#!/usr/bin/env python3 -u
"""
Vectorized Env + Segmented Path RL — proper "trade when" training.
  B envs || runs L-step episodes in parallel.
  
Env tracks: latent[t], w_prev (position state)
Policy: delta-based soft threshold → learns when to enter/hold/exit.

Architecture:
- B=256 parallel environments
- L=64 steps per episode (≈16h of 15m data)
- Each bar: all B envs step in parallel (batched forward)
- Each episode: L consecutive bars sampled from training data
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
# Use CPU for training (tiny model, MPS kernel overhead dominates)
TRAIN_DEV = 'cpu'
EVAL_DEV = DEV  # eval can stay on MPS
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

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)
# CPU copies for training
lt_cpu = lt.cpu()
rt_cpu = rt.cpu()

# ── Vectorized Environment ──
class VecEnv:
    """B parallel environments. Each env: w_prev + latent → step."""
    def __init__(self, B=256, L=64, device='cpu'):
        self.B = B
        self.L = L
        self.device = device
        self.lt = lt_cpu if device == 'cpu' else lt
        self.rt = rt_cpu if device == 'cpu' else rt
        self.reset()
    
    def reset(self, split='train'):
        """Sample B new episodes from training data."""
        lo, hi = tr if split == 'train' else (va if split == 'val' else te)
        self.starts = np.random.randint(lo, hi - self.L - 1, size=self.B)
        self.w = torch.zeros(self.B, NC, device=self.device)
        self.t = 0
        self.split = split
        return self._get_obs()
    
    def _get_obs(self):
        """Current observation for each env: latent[t] + w_prev."""
        ids = torch.from_numpy(self.starts + self.t).to(self.device)
        z = self.lt[ids]
        return z, self.w
    
    def step(self, w_new):
        """Take action w_new, get reward. w_new: (B, NC)"""
        ids = torch.from_numpy(self.starts + self.t).to(self.device)
        r = self.rt[ids]
        pr = (w_new * r).sum(1)
        to = (w_new - self.w).abs().sum(1)
        self.w = w_new
        self.t += 1
        done = (self.t >= self.L)
        return pr - FEE * to, done
    
    def roll_out(self, policy):
        """Run full episode for all envs. Returns all rewards."""
        self.reset(self.split)
        all_returns = []
        for _ in range(self.L):
            z, w_prev = self._get_obs()
            w_new = policy(z, w_prev)
            r, d = self.step(w_new)
            all_returns.append(r)
        return torch.stack(all_returns, dim=1)

# ── Policy: Delta + Soft Threshold ──
class DeltaSoftThresh(nn.Module):
    """w[t] = w[t-1] + soft_threshold(score([latent, w_prev]), theta)"""
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        D_in = DZ + NC
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)
        self.thresh_head = nn.Linear(H, NC)
        nn.init.constant_(self.thresh_head.bias, -2.0)  # θ≈0.12 init
    def forward(self, z, w_prev):
        x = torch.cat([z, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        abs_s = score.abs()
        delta = score.sign() * (abs_s - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)

# ── Training ──
def train(pi, name, B=256, L=64, STEPS=10000, lr=3e-4):
    print(f'\n{"="*60}')
    print(f'[Train] {name}  B={B}  L={L}')
    print(f'{"="*60}')
    pi.to(TRAIN_DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    
    env = VecEnv(B=B, L=L, device=TRAIN_DEV)
    best_sr = -10
    t0 = time.time()
    
    for step in range(STEPS):
        # Roll out B envs × L steps
        returns = env.roll_out(pi)  # (B, L)
        
        # Sharpe across all B×L portfolio returns
        pr_flat = returns.reshape(-1)
        loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        
        if step % 2500 == 0 or step == 0:
            pi.to(EVAL_DEV)
            sr = eval_fast(pi)
            best_sr = max(best_sr, sr)
            pi.to(TRAIN_DEV)
            print(f'  Step {step:>5d}  Val SR={sr:.2f}  wall={time.time()-t0:.0f}s')
    
    pi.to(EVAL_DEV)
    s, t, n, a = eval_full(pi, 'test')
    print(f'  [{name}] Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:.2f}  Act={a:.2%}')
    return {'test': s, 'to': t, 'net': n, 'active': a}

# ── Evaluation (2-pass fixed-point, fast) ──
@torch.no_grad()
def eval_fast(pi, split='val'):
    """2-pass fixed-point. Much faster than sequential, good enough for tracking."""
    lo, hi = va if split == 'val' else te
    z = lt[lo:hi]; n = hi - lo
    # Pass 1: w ~ f(z, zeros)
    w = pi(z, torch.zeros(n, NC, device=DEV))
    # Pass 2: refine with shifted w
    wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
    w = pi(z, wp)
    # Pass 3: more refine
    wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
    w = pi(z, wp)
    pr = (w * rt[lo:hi]).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    return sr

@torch.no_grad()
def eval_full(pi, split='test'):
    """Full evaluation with weights for analysis. Sequential (slow but accurate)."""
    lo, hi = va if split == 'val' else te
    z = lt[lo:hi]; n = hi - lo
    w = torch.zeros(1, NC, device=DEV)
    pr, ws = [], []
    for i in range(n):
        w = pi(z[i:i+1], w)
        ws.append(w)
        pr.append((w * rt[lo+i:lo+i+1]).sum().item())
    w_np = torch.cat(ws, dim=0).cpu().numpy()
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w_np, axis=0)).sum(1).mean()
    fee_cost = FEE * np.abs(np.diff(w_np, axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    sf = (pr - fee_cost).mean() / max((pr - fee_cost).std(), 1e-8) * ANNUAL
    act = (w_np.sum(1) > 0.001).mean()
    return sr, to, sf, act

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

print('═══ Vectorized Env RL — "Trade When" ═══')

# 1. Delta soft threshold (θ=0.15)
results['ds_015'] = train(DeltaSoftThresh(theta_max=0.15), 'DeltaSoft θ=0.15')

# 2. Different threshold
results['ds_010'] = train(DeltaSoftThresh(theta_max=0.10), 'DeltaSoft θ=0.10')

# 3. Different threshold
results['ds_020'] = train(DeltaSoftThresh(theta_max=0.20), 'DeltaSoft θ=0.20')

# ── Summary ──
print(f'\n{"="*60}')
print('FINAL SUMMARY')
print(f'{"="*60}')
print(f'{"Method":<25s} {"Test SR":>7s} {"TO":>7s} {"Net SR":>7s} {"Active":>7s}')
print('-' * 56)
for name in sorted(results.keys()):
    r = results[name]
    print(f'{name:<25s} {r["test"]:>7.2f} {r["to"]:>7.4f} {r["net"]:>7.2f} {r["active"]:>7.2%}')

# Compare with previous best
print(f'\n{"—"*56}')
print(f'{"Prev Best (SoftThresh θ=0.3)":<25s} {"0.94":>7s} {"0.003":>7s} {"0.79":>7s} {"—":>7s}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
