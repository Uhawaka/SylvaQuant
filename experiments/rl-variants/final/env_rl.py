#!/usr/bin/env python3 -u
"""
DeltaSoft RL Environment — Gym API + Differentiable VecEnv + Policy.

Best config (ChunkEnv B=1024, L=256, 300 steps, seed=1111):
  Test SR=0.85, Net SR=+0.77, TO=0.014 (8x lower than VecEnv!)

Architecture:
  DeltaSoftThresh (delta-based soft threshold policy)
    └── ChunkEnv (B parallel envs, each on a DIFFERENT contiguous chunk)
          └── Direct Sharpe optimization (w·ret - fee·|Δw|)
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Constants ──
NC = 9     # number of assets
DZ = 16    # latent dimension
H = 16     # hidden size
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004

# ══════════════════════════════════════════════
# Policy: Delta Soft Threshold
# ══════════════════════════════════════════════
class DeltaSoftThresh(nn.Module):
    """
    Delta-based soft threshold policy.
    
    Core idea: w[t] = clamp(w[t-1] + delta[t], -1, 1)
    
    where delta[t] = sign(score)·max(|score| - θ, 0)
    
    Natural behavior:
      score ≈ 0     → delta ≈ 0   → HOLD position
      score > θ     → delta > 0   → INCREASE position (enter/add)
      score < -θ    → delta < 0   → DECREASE position (exit/reduce)
    
    θ is learned per-asset per-bar via a separate head:
      θ = sigmoid(thresh_head(latent, w_prev)) * θ_max
    
    This is equivalent to L1-regularized optimal control:
      min_Δw  -(Δw·score) + θ·|Δw|
      → Δw = soft_threshold(score, θ)
    """
    def __init__(self, theta_max=0.18):
        super().__init__()
        self.theta_max = theta_max
        D_in = DZ + NC  # latent + w_prev
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)   # raw signal score
        self.thresh_head = nn.Linear(H, NC)  # learns how selective to be
        nn.init.constant_(self.thresh_head.bias, -2.0)  # init θ ≈ 0.12

    def forward(self, z, w_prev):
        """
        z: (B, DZ) latent features
        w_prev: (B, NC) previous positions
        returns: (B, NC) new positions in [-1, 1]
        """
        x = torch.cat([z, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)


# ══════════════════════════════════════════════
# Vectorized Environment (Differentiable)
# ══════════════════════════════════════════════
class VecEnv:
    """
    B parallel environments, fully differentiable.
    
    Each episode: L consecutive bars from training data.
    State: [latent[t], w_prev]
    Action: w_new ∈ [-1, 1]^9
    Reward: w·ret - fee·|Δw|
    
    Gradients flow through ALL L×B steps for end-to-end Sharpe optimization.
    """
    def __init__(self, lt, rt, tr, B=256, L=32):
        self.lt = lt
        self.rt = rt
        self.tr = tr
        self.B = B
        self.L = L
        
    def reset(self):
        lo, hi = self.tr
        self.starts = np.random.randint(lo, hi - self.L - 1, size=self.B)
        self.w = torch.zeros(self.B, NC)
        self.t = 0
        
    def roll_out(self, pi):
        """Run full L-step episode for all B envs. Returns (B, L) returns tensor.
        
        Optimized: pre-gather all indices before the L-loop (single contiguous gather),
        then use strided slices inside the loop (O(1) each).
        """
        self.reset()
        # Pre-gather all indices — single contiguous gather
        idx = (torch.from_numpy(self.starts).long().unsqueeze(1) 
               + torch.arange(self.L))
        z_all = self.lt[idx]  # (B, L, DZ)
        r_all = self.rt[idx]  # (B, L, NC)
        
        w = torch.zeros(self.B, NC)
        rets = []
        for t in range(self.L):
            w_new = pi(z_all[:, t], w)        # strided slice, O(1)
            pr = (w_new * r_all[:, t]).sum(1)   # (B,)
            to = (w_new - w).abs().sum(1)       # (B,)
            rets.append(pr - FEE * to)
            w = w_new
        return torch.stack(rets, dim=1)


# ══════════════════════════════════════════════
# Parallel Chunk Environment (Continuous)
# ══════════════════════════════════════════════
class ChunkEnv:
    """
    B parallel envs, each on a DIFFERENT contiguous time chunk.
    All step forward in sync — no random resets mid-episode.
    
    Unlike VecEnv which resets to random positions every step,
    ChunkEnv gives each env a long continuous sequence where:
      - w_prev carries over naturally (no sudden exits)
      - Turnover accumulates realistically
      - The policy learns long-term consequences
    
    Best config: B=1024, L=256, ~300 steps (same data as 10K VecEnv steps)
      → Test SR=0.85, TO=0.014, Net SR=+0.77 (vs VecEnv TO=0.115)
    """
    def __init__(self, lt, rt, tr, B=1024, L=256):
        self.lt = lt
        self.rt = rt
        self.lo, self.hi = tr
        self.B = B
        self.L = L
        self.max_start = max(0, self.hi - self.lo - L)
    
    def roll_out(self, pi):
        """Each env runs L steps on its own contiguous chunk."""
        starts = np.random.randint(0, self.max_start + 1, size=self.B) + self.lo
        idx = (torch.from_numpy(starts).long().unsqueeze(1) 
               + torch.arange(self.L))
        z_all = self.lt[idx]  # (B, L, DZ)
        r_all = self.rt[idx]  # (B, L, NC)
        
        w_prev = torch.zeros(self.B, NC)
        returns = []
        for t in range(self.L):
            w_new = pi(z_all[:, t], w_prev)
            pr = (w_new * r_all[:, t]).sum(1)
            to = (w_new - w_prev).abs().sum(1)
            returns.append(pr - FEE * to)
            w_prev = w_new
        return torch.stack(returns, dim=1)


# ══════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════
@torch.no_grad()
def eval_sequential(pi, lt, rt, lo, hi):
    """
    Full sequential evaluation. Accurate but slower.
    Processes bars one at a time, tracking w_prev.
    
    Returns: (sr, to, net_sr, active_ratio)
    """
    z = lt[lo:hi]; n = hi - lo
    w = torch.zeros(1, NC)
    pr, ws = [], []
    for i in range(n):
        w = pi(z[i:i+1], w)
        ws.append(w)
        pr.append((w * rt[lo+i:lo+i+1]).sum().item())
    w_np = torch.cat(ws, dim=0).numpy()
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w_np, axis=0)).sum(1).mean()
    fee_cost = np.concatenate([[0.0], FEE * np.abs(np.diff(w_np, axis=0)).sum(1)])
    net_sr = (pr - fee_cost).mean() / max((pr - fee_cost).std(), 1e-8) * ANNUAL
    active = (w_np.sum(1) > 0.001).mean()
    return sr, to, net_sr, active


@torch.no_grad()
def eval_fast(pi, lt, rt, lo, hi, bs=1024):
    """
    Batch-sequential eval for fast tracking during training.
    Processes BS bars at a time, tracking w_prev sequentially within each chunk.
    """
    z_all = lt[lo:hi]; r_all = rt[lo:hi]
    n = hi - lo
    w = torch.zeros(1, NC)
    all_pr = []
    for i in range(0, n, bs):
        end = min(i + bs, n)
        z_b = z_all[i:end]
        r_b = r_all[i:end]
        ws = []
        for t in range(end - i):
            w = pi(z_b[t:t+1], w)
            ws.append(w)
        w_b = torch.cat(ws, dim=0)
        all_pr.append((w_b * r_b).sum(1))
    pr = torch.cat(all_pr).cpu().numpy()
    return pr.mean() / max(pr.std(), 1e-8) * ANNUAL


# ══════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════
def train(pi, lt, rt, tr_idx, va_idx, te_idx,
          B=1024, L=256, STEPS=300, lr=3e-4, seed=1111, verbose=True,
          env_type='chunk'):
    """
    Train a DeltaSoftThresh policy with end-to-end Sharpe optimization.
    
    Args:
        pi: DeltaSoftThresh policy
        lt: latent tensor (N, DZ)
        rt: return tensor (N, NC)
        tr_idx: (lo, hi) training range
        va_idx: (lo, hi) validation range
        te_idx: (lo, hi) test range
        B: parallel envs
        L: episode length (continuous bars per env per step)
        STEPS: training steps
        lr: learning rate
        seed: random seed
        env_type: 'chunk' (ChunkEnv, default) or 'vec' (VecEnv)
    
    Returns:
        dict with test metrics
    """
    torch.manual_seed(seed); np.random.seed(seed)
    
    if env_type == 'chunk':
        env = ChunkEnv(lt, rt, tr_idx, B=B, L=L)
    else:
        env = VecEnv(lt, rt, tr_idx, B=B, L=L)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    
    # For fast eval
    va_lo, va_hi = va_idx
    
    for step in range(STEPS):
        rets = env.roll_out(pi)
        pr_flat = rets.reshape(-1)
        loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        
        if verbose and step > 0 and step % 5000 == 0:
            sr = eval_fast(pi, lt, rt, va_lo, va_hi)
            print(f'  Step {step:>5d}  Val SR={sr:.2f}')
    
    # Final full evaluation
    if verbose:
        print('  Final eval...')
    s, to, net, act = eval_sequential(pi, lt, rt, te_idx[0], te_idx[1])
    sv, _, nv, _ = eval_sequential(pi, lt, rt, va_idx[0], va_idx[1])
    
    if verbose:
        print(f'  Test: SR={s:.2f}  TO={to:.4f}  Net SR={net:+.2f}  Act={act:.2%}')
        print(f'  Val:  SR={sv:.2f}  Net SR={nv:+.2f}')
    
    return {'test_sr': s, 'test_to': to, 'test_net': net, 'test_act': act,
            'val_sr': sv, 'val_net': nv}


def save_policy(pi, path, theta_max=None):
    """Save a DeltaSoftThresh policy checkpoint for online inference."""
    if theta_max is None:
        theta_max = pi.theta_max
    torch.save({
        'model_state': pi.state_dict(),
        'theta_max': theta_max,
        'n_coins': NC,
        'latent_dim': DZ,
    }, path)
    print(f'  Saved policy to {path}')


# ══════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════
def load_data(path='data/rl_exp/exp_data.npz'):
    """Load and normalize RL experiment data."""
    import numpy as np
    d = np.load(path)
    lat = d['latents'].astype(np.float32)
    ret = d['raw_ret'].astype(np.float32)
    tr = d['train_idx']; va = d['val_idx']; te = d['test_idx']
    
    # Normalize latents on training stats
    lm = lat[:tr[1]].mean(0, keepdims=True)
    ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
    lat_n = ((lat - lm) / ls).astype(np.float32)
    
    return (torch.from_numpy(lat_n).float(),
            torch.from_numpy(ret).float(),
            tr, va, te)


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
if __name__ == '__main__':
    import time
    ROOT = Path(__file__).resolve().parent.parent.parent.parent
    data_path = str(ROOT / 'data/rl_exp/exp_data.npz')
    
    print('═══ DeltaSoft RL Training (ChunkEnv) ═══')
    lt, rt, tr, va, te = load_data(data_path)
    print(f'Data: {lt.shape[0]} bars, train={tr}, val={va}, test={te}')
    print(f'ChunkEnv: B=1024, L=256, 300 steps')
    
    pi = DeltaSoftThresh(theta_max=0.18)
    t0 = time.time()
    result = train(pi, lt, rt, tr, va, te,
                   B=1024, L=256, STEPS=300, seed=1111)
    print(f'Time: {time.time()-t0:.0f}s')
    
    # Sweep θ_max with ChunkEnv
    print('\n─── θ_max sweep (ChunkEnv) ───')
    for theta in [0.12, 0.15, 0.18, 0.20]:
        pi = DeltaSoftThresh(theta_max=theta)
        r = train(pi, lt, rt, tr, va, te,
                  B=1024, L=256, STEPS=300, seed=1111, verbose=False)
        print(f'  θ={theta:.2f}: SR={r["test_sr"]:.2f}  Net SR={r["test_net"]:+.2f}  TO={r["test_to"]:.4f}')
