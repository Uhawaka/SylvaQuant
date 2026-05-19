#!/usr/bin/env python3 -u
"""
DeltaSoft — Optimized Continuous Chunk Training on MPS.

Key optimizations over VecEnv:
  1. MPS device (GPU)
  2. Pre-sliced data: (N_seg, L, DZ) → single gather per step
  3. No torch indexing inside L-loop (just strided slices)
  4. Sequential w_prev tracking is preserved
"""
import sys, warnings, time, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import torch, torch.nn as nn

# ── Config ──
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004
SEED = 1111
THETA = 0.18
L = 32
B = 2048   # larger batch for GPU
STEPS = 5000  # more steps, faster per step

# ── Data ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

# Normalize latents
lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

# Pre-slice training data into segments: (N_seg, L, DZ)
train_lo, train_hi = tr
train_data = lat_n[train_lo:train_hi]   # (T_train, DZ)
train_ret = ret[train_lo:train_hi]       # (T_train, NC)
T_train = len(train_data)
N_seg = T_train // L
S_tr = torch.from_numpy(train_data[:N_seg*L].reshape(N_seg, L, DZ)).to(DEV, non_blocking=True)
R_tr = torch.from_numpy(train_ret[:N_seg*L].reshape(N_seg, L, NC)).to(DEV, non_blocking=True)

# Eval tensors
@torch.no_grad()
def _to_dev(x): return torch.from_numpy(x).to(DEV)

S_va = _to_dev(lat_n[va[0]:va[1]])
R_va = _to_dev(ret[va[0]:va[1]])
S_te = _to_dev(lat_n[te[0]:te[1]])
R_te = _to_dev(ret[te[0]:te[1]])

print(f'Device: {DEV}')
print(f'Train segments: {N_seg} x L={L} = {N_seg*L} bars')
print(f'Val: {va[1]-va[0]} bars, Test: {te[1]-te[0]} bars')

# ── Policy ──
class DeltaSoftThresh(nn.Module):
    def __init__(self, theta_max=THETA):
        super().__init__()
        self.theta_max = theta_max
        self.encoder = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)
        self.thresh_head = nn.Linear(H, NC)
        nn.init.constant_(self.thresh_head.bias, -2.0)
    def forward(self, z, w_prev):
        x = torch.cat([z, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)

# ── Fast training ──
def train_fast(pi, STEPS=5000, B=2048, L=32, verbose=True):
    """
    Optimized training loop.
    
    Data is pre-sliced into (N_seg, L, *). Each step:
      1. Sample B segments → (B, L, *) [single contiguous gather]
      2. Run L-step rollout with strided slices [O(1) per step]
      3. Compute Sharpe, backward
    
    No torch indexing in the hot loop. Pure batch ops on GPU.
    """
    pi.to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=3e-4, weight_decay=1e-5)
    t0 = time.time()
    
    for step in range(STEPS):
        # Sample B random segments — single gather on GPU
        perm = torch.randint(0, N_seg, (B,), device=DEV)
        z_seg = S_tr[perm]   # (B, L, DZ)
        r_seg = R_tr[perm]   # (B, L, NC)
        
        # Rollout with w_prev tracking
        w = torch.zeros(B, NC, device=DEV)
        rets = []
        for t in range(L):
            w_new = pi(z_seg[:, t], w)     # (B, NC) — strided slice, O(1)
            pr = (w_new * r_seg[:, t]).sum(1)  # (B,)
            to = (w_new - w).abs().sum(1)      # (B,)
            rets.append(pr - FEE * to)
            w = w_new
        
        ret_tensor = torch.stack(rets, dim=1).reshape(-1)  # (B*L,)
        loss = -ret_tensor.mean() / (ret_tensor.std() + 1e-8)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        
        if verbose and (step % 2500 == 0 or step == 0):
            s = eval_fast(pi)
            print(f'  Step {step:>5d}  Val SR={s:.2f}  wall={time.time()-t0:.0f}s')
    
    s, to, net, act = eval_full(pi, 'test')
    sv, _, nv, _ = eval_full(pi, 'val')
    if verbose:
        print(f'  Test: SR={s:.2f}  TO={to:.4f}  Net SR={net:+.2f}  Act={act:.2%}')
        print(f'  Val:  SR={sv:.2f}  Net SR={nv:+.2f}')
        print(f'  Time: {time.time()-t0:.0f}s')
    
    return {'test_sr': s, 'test_to': to, 'test_net': net, 'test_act': act, 'val_sr': sv, 'val_net': nv}

# ── Eval ──
@torch.no_grad()
def eval_fast(pi):
    """Batch-sequential eval on validation set."""
    n = S_va.shape[0]; w = torch.zeros(1, NC, device=DEV)
    bs = 2048; all_pr = []
    for i in range(0, n, bs):
        end = min(i + bs, n)
        z_b, r_b = S_va[i:end], R_va[i:end]
        ws = []
        for t in range(end - i):
            w = pi(z_b[t:t+1], w)
            ws.append(w)
        all_pr.append((torch.cat(ws, dim=0) * r_b).sum(1))
    pr = torch.cat(all_pr).cpu().numpy()
    return pr.mean() / max(pr.std(), 1e-8) * ANNUAL

@torch.no_grad()
def eval_full(pi, split='test'):
    z, r = (S_te, R_te) if split == 'test' else (S_va, R_va)
    n = z.shape[0]
    w = torch.zeros(1, NC, device=DEV)
    pr, ws = [], []
    for i in range(n):
        w = pi(z[i:i+1], w)
        ws.append(w); pr.append((w * r[i:i+1]).sum().item())
    w_np = torch.cat(ws, dim=0).cpu().numpy()
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w_np, axis=0)).sum(1).mean()
    fee = np.concatenate([[0.0], FEE * np.abs(np.diff(w_np, axis=0)).sum(1)])
    net = (pr - fee).mean() / max((pr - fee).std(), 1e-8) * ANNUAL
    act = (w_np.sum(1) > 0.001).mean()
    return sr, to, net, act

# ── Run ──
torch.manual_seed(SEED); np.random.seed(SEED)

print('\n═══ Optimized DeltaSoft Training ═══')
print(f'B={B}, L={L}, θ={THETA}, seed={SEED}')
print()

# Warmup: tiny run to compile/init MPS
print('Warmup...')
pi_w = DeltaSoftThresh()
train_fast(pi_w, STEPS=100, B=256, verbose=False)
print('Warmup done.\n')

# Main training
pi = DeltaSoftThresh()
result = train_fast(pi, STEPS=STEPS, B=B)

# ── Sweep B for comparison ──
print('\n─── B sweep ───')
for B_test in [512, 1024, 4096]:
    torch.manual_seed(SEED); np.random.seed(SEED)
    pi = DeltaSoftThresh()
    r = train_fast(pi, STEPS=5000, B=B_test, verbose=False)
    print(f'  B={B_test:>5d}: SR={r["test_sr"]:.2f}  Net SR={r["test_net"]:+.2f}  TO={r["test_to"]:.4f}')

print(f'\nDone.')
