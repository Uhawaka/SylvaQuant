#!/usr/bin/env python3 -u
"""
Proper RL Environment Design — Gym API + Differentiable VecEnv.

Architecture:
  Gym Env (numpy, clean, testable)
    └── TorchVecEnv (PyTorch, differentiable, fast)
          └── Policy (learns w = soft_threshold(score))

TorchVecEnv preserves gradient flow for Diff Sharpe optimization.
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'cpu'
NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004

# ── Data ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

# PyTorch tensors (keep on CPU for tiny model)
lt = torch.from_numpy(lat_n).float()
rt = torch.from_numpy(ret).float()

# ═══════════════════════════════════════════
# Layer 1: Gym environment (numpy, clean)
# ═══════════════════════════════════════════
class PortfolioGymEnv(gym.Env):
    """
    Gym portfolio env. Single episode = L consecutive bars.
    State: [latent[t], w_prev] (25-dim)
    Action: w_new ∈ [-1, 1]^9
    Reward: w·ret - fee·|Δw|
    """
    metadata = {"render_modes": []}
    
    def __init__(self, latents=lat_n, returns=ret, L=64, fee=FEE, id_range=None):
        super().__init__()
        self.latents = latents
        self.returns = returns
        self.L = L
        self.fee = fee
        self.N = len(latents)
        self.id_range = id_range or (0, self.N - L)
        
        self.observation_space = spaces.Box(-np.inf, np.inf, (DZ + NC,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (NC,), np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'idx' in options:
            self.idx = options['idx']
        else:
            lo, hi = self.id_range
            self.idx = self.np_random.integers(lo, hi)
        self.t = 0
        self.w_prev = np.zeros(NC, np.float32)
        return self._obs(), {}
    
    def step(self, action):
        w = np.clip(action, -1, 1).astype(np.float32)
        r = self.returns[self.idx + self.t]
        pr = w @ r
        to = np.abs(w - self.w_prev).sum()
        reward = pr - self.fee * to
        self.w_prev = w
        self.t += 1
        done = self.t >= self.L
        return self._obs(), reward.item() if hasattr(reward, 'item') else reward, done, False, {}
    
    def _obs(self):
        return np.concatenate([self.latents[self.idx + self.t], self.w_prev]).astype(np.float32)


# ═══════════════════════════════════════════
# Layer 2: Differentiable VecEnv (PyTorch)
# ═══════════════════════════════════════════
class TorchVecEnv:
    """
    B parallel envs, fully differentiable (autograd flows through L steps).
    
    Usage:
        env = TorchVecEnv(B=256, L=64)
        obs = env.reset()                         # (B, 25) 
        action, ret, obss = env.step(obs, policy)  # roll out L steps w/ grad
        loss = -sharpe(ret)
        loss.backward()                           # gradients through policy!
    """
    def __init__(self, B=256, L=64, data_range=None):
        self.B = B
        self.L = L
        self.N = lt.shape[0]
        lo, hi = data_range or (tr[0], tr[1] - L)
        self.data_lo = lo
        self.data_hi = hi
    
    def reset(self):
        """Start new episodes: random starts, zero positions."""
        self.starts = torch.randint(self.data_lo, self.data_hi, (self.B,))
        self.t = 0
        self.w_prev = torch.zeros(self.B, NC, device=DEV)
        return self._obs()
    
    def _obs(self):
        """Current observation for each env: [latent[t], w_prev]."""
        z = lt[self.starts + self.t]  # (B, DZ)
        return torch.cat([z, self.w_prev], dim=1)  # (B, 25)
    
    def roll(self, policy, n_steps=1, store=True):
        """Step all B envs for n_steps, return actions & returns with gradients."""
        w_prev = self.w_prev
        returns = []
        
        for step in range(n_steps):
            z = lt[self.starts + self.t]  # (B, DZ)
            obs = torch.cat([z, w_prev], dim=1)
            
            w_new = policy(obs)  # (B, NC) — gradients flow!
            r = rt[self.starts + self.t]
            
            pr = (w_new * r).sum(1)
            to = (w_new - w_prev).abs().sum(1)
            ret = pr - FEE * to
            
            returns.append(ret)
            w_prev = w_new
            self.t += 1
        
        self.w_prev = w_prev.detach()  # detach for next call
        self._returns = returns
        
        if store:
            self._obs_store = obs  # for eval
        return returns, w_prev


# ═══════════════════════════════════════════
# Policy: Delta Soft Threshold
# ═══════════════════════════════════════════
class DeltaSoftThresh(nn.Module):
    """
    w[t] = clamp(w[t-1] + delta[t], -1, 1)
    delta[t] = sign(s) · max(|s| - θ, 0)  (soft threshold on CHANGE)
    
    Natural behavior:
      When score ≈ 0: delta ≈ 0 → w[t] ≈ w[t-1] (HOLD)
      When score > θ: delta > 0 → increase position (ENTER/ADD)
      When score < θ: delta < 0 → decrease position (EXIT/REDUCE)
    """
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        D_in = DZ + NC  # 25
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)   # raw score for delta direction
        self.thresh_head = nn.Linear(H, NC)  # threshold for delta sparsity
        nn.init.constant_(self.thresh_head.bias, -2.0)  # θ≈0.12 init
    
    def forward(self, obs):
        """obs: (B, 25) = [latent(16), w_prev(9)]"""
        w_prev = obs[:, DZ:DZ+NC]  # last 9 dims = previous weight
        h = self.encoder(obs)
        score = self.score_head(h)  # unbounded score
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max  # [0, θ_max]
        
        # Delta = soft threshold on score (what to CHANGE)
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        
        # New weight = old weight + delta, clipped to [-1, 1]
        w = (w_prev + delta).clamp(-1, 1)
        return w


# ═══════════════════════════════════════════
# Layer 3: Training
# ═══════════════════════════════════════════
def train(policy, B=256, L=64, STEPS=10000, lr=3e-4):
    print(f'\n{"="*60}')
    print(f'TRAIN  B={B}  L={L}  STEPS={STEPS}')
    print(f'{"="*60}')
    
    env = TorchVecEnv(B=B, L=L)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    
    for step in range(STEPS):
        env.reset()
        returns, _ = env.roll(policy, n_steps=L)
        
        # Compute Sharpe (grad flows through all L×B returns)
        ret_tensor = torch.stack(returns, dim=1).reshape(-1)  # (B×L,)
        mu = ret_tensor.mean()
        sd = ret_tensor.std() + 1e-8
        loss = -mu / sd
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        
        if step % 2500 == 0 or step == 0:
            sr_val, _, _ = eval_seq(policy, 'val')
            print(f'  Step {step:>5d}  Val SR={sr_val:.2f}  '
                  f'loss={loss.item():+.4f}  wall={time.time()-t0:.0f}s')
    
    s, n, t, a = eval_full(policy)
    print(f'  Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={s:.2f}  Net SR={n:.2f}  TO={t:.4f}  Act={a:.2%}')
    return {'test_sr': s, 'net': n, 'to': t, 'active': a}


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════
@torch.no_grad()
def eval_seq(policy, split='val'):
    """Sequential eval (accurate). Uses Gym env for clean interface."""
    lo, hi = va if split == 'val' else te
    n = hi - lo
    w = torch.zeros(1, NC)
    pr = []
    for i in range(n):
        z = lt[lo + i:lo + i + 1]
        obs = torch.cat([z, w], dim=1)
        w = policy(obs)
        r = rt[lo + i:lo + i + 1]
        pr.append((w * r).sum().item())
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    return sr, 0, 0

@torch.no_grad()
def eval_full(policy, split='test'):
    lo, hi = te if split == 'test' else va
    n = hi - lo
    w = torch.zeros(1, NC)
    ws, pr = [], []
    for i in range(n):
        z = lt[lo + i:lo + i + 1]
        obs = torch.cat([z, w], dim=1)
        w = policy(obs)
        ws.append(w)
        pr.append((w * rt[lo + i:lo + i + 1]).sum().item())
    w_np = torch.cat(ws, dim=0).cpu().numpy()
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w_np, axis=0)).sum(1).mean()
    fee_cost = FEE * np.abs(np.diff(w_np, axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    net_sr = (pr - fee_cost).mean() / max((pr - fee_cost).std(), 1e-8) * ANNUAL
    act = (w_np.sum(1) > 0.001).mean()
    return sr, net_sr, to, act


# ═══════════════ RUN ═══════════════
if __name__ == '__main__':
    print('═══ RL Environment — Gym + Diffable VecEnv ═══')
    
    # Test Gym env
    print('\n[Gym Env Test]')
    env = PortfolioGymEnv(L=5)
    obs, _ = env.reset()
    print(f'  obs.shape={obs.shape}  action.shape={env.action_space.shape}')
    total_r = 0
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, d, _, _ = env.step(a)
        total_r += r
    print(f'  episode return={total_r:.6f}')
    
    # Test TorchVecEnv
    print('\n[TorchVecEnv Test]')
    venv = TorchVecEnv(B=4, L=5)
    obs = venv.reset()
    pi_test = DeltaSoftThresh()
    returns, _ = venv.roll(pi_test, n_steps=5)
    print(f'  returns: {[f"{r.mean().item():.4f}" for r in returns]}')
    
    # Sweep: theta_max
    for tm in [0.10, 0.20, 0.30]:
        print(f'\n[θ={tm:.2f} B=256 L=32 10K steps]')
        pi = DeltaSoftThresh(theta_max=tm)
        result = train(pi, B=256, L=32, STEPS=10000)
