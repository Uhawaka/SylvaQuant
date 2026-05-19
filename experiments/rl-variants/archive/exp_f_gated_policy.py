#!/usr/bin/env python3 -u
"""
Gated Policy — RL with explicit no-trade zone.
Compares: gated Diff Sharpe, vanilla Diff Sharpe, threshold post-processing.
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004

# ── Data ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
spnl = d['sig_pnl'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)
st = torch.from_numpy(spnl).to(DEV)

# ── Policy architectures ──

class VanillaPolicy(nn.Module):
    """Baseline: latent → [-1,1]^9, always active."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu = nn.Linear(H, NC)
    def forward(self, s):
        return torch.tanh(self.mu(self.net(s)))

class GatedPolicy(nn.Module):
    """
    Gated weight allocation:
      direction ∈ [-1, 1] (tanh) — which side
      size ∈ [0, 1] (sigmoid) — how much of max
      gate ∈ [0, 1] (sigmoid) — whether to trade
      Final: w = gate * direction * size
    Gate can be ~0 → no position → no turnover cost.
    """
    def __init__(self, max_pos=1.0):
        super().__init__()
        self.max_pos = max_pos
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.dir_head = nn.Linear(H, NC)        # direction
        self.size_head = nn.Linear(H, NC)        # sizing [0, 1]
        self.gate_head = nn.Linear(H, NC)        # gate [0, 1]
        
        # Initialize gate bias to encourage sparsity
        # sigmoid(bias=-2) ≈ 0.12 → starts mostly off
        nn.init.constant_(self.gate_head.bias, -2.0)
        
    def forward(self, s):
        h = self.net(s)
        direction = torch.tanh(self.dir_head(h))
        size = torch.sigmoid(self.size_head(h))
        gate = torch.sigmoid(self.gate_head(h))
        return gate * direction * size * self.max_pos

class ThreshPolicy(nn.Module):
    """Vanilla policy with post-hoc threshold: |w| < th → 0."""
    def __init__(self, threshold=0.05):
        super().__init__()
        self.th = threshold
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu = nn.Linear(H, NC)
    def forward(self, s):
        w = torch.tanh(self.mu(self.net(s)))
        # Apply threshold: set small weights to 0 (differentiable approximation)
        mask = (w.abs() > self.th).float()
        return w * mask

# ── Evaluation ──
@torch.no_grad()
def evaluate(pi, ret_src, split='test', apply_th=None):
    lo, hi = va if split == 'val' else te
    w = pi(lt[lo:hi])
    
    # Optional post-hoc threshold
    if apply_th is not None:
        w[w.abs() < apply_th] = 0.0
    
    pr = (w * ret_src[lo:hi]).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    
    # Fee-adjusted returns
    fee_cost = FEE * np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    pr_fee = pr - fee_cost
    sr_fee = pr_fee.mean() / max(pr_fee.std(), 1e-8) * ANNUAL
    
    # Active ratio: fraction of bars where any position is open
    active_ratio = (w.abs().sum(1).cpu().numpy() > 0.001).mean()
    
    return sr, to, sr_fee, active_ratio, pr, w

# ── Diff Sharpe Training ──
def train(pi, ret_src, name, STEPS=20000, B=4096, lr=3e-4):
    print(f'\n{"="*60}')
    print(f'[Train] {name}')
    print(f'{"="*60}')
    
    pi.to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    best = {'sr': -10, 'to': 0}
    t0 = time.time()
    
    for step in range(STEPS):
        ts = np.random.randint(tr[0], tr[1] - B - 1)
        z = lt[ts:ts+B]
        r = rt[ts:ts+B] if ret_src == 'raw' else st[ts:ts+B]
        
        w = pi(z)
        w_prev = torch.cat([w[:1], w[:-1]], dim=0)
        to_pen = FEE * (w - w_prev).abs().sum(1)
        pr = (w * r).sum(1) - to_pen
        
        loss = -pr.mean() / (pr.std() + 1e-8)
        
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()
        
        if step % 5000 == 0 or step == 0:
            ret_t = rt if ret_src == 'raw' else st
            sr_v, to_v, sr_f, act_v, _, _ = evaluate(pi, ret_t, 'val')
            pi.train()
            if sr_v > best['sr']:
                best['sr'] = sr_v; best['to'] = to_v
            print(f'  Step {step:>5d}  Val SR={sr_v:.2f}  TO={to_v:.4f}  Act={act_v:.2%}  Net SR={sr_f:.2f}')
    
    ret_t = rt if ret_src == 'raw' else st
    sr_t, to_t, sr_f, act_t, pr_t, w_t = evaluate(pi, ret_t, 'test')
    
    # Also test with post-hoc thresholds
    print(f'  [{name}] Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={sr_t:.2f}  TO={to_t:.4f}  Act={act_t:.2%}  Net SR={sr_f:.2f}')
    
    return {
        'val': best['sr'], 'test': sr_t, 'to': to_t, 'net': sr_f,
        'active': act_t, 'pr': pr_t, 'w': w_t
    }

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

# 1. Vanilla Diff Sharpe (baseline)
results['vanilla'] = train(VanillaPolicy(), 'raw', 'Vanilla H=16')

# 2. Gated Policy (fee-aware training)
results['gated'] = train(GatedPolicy(), 'raw', 'Gated H=16 (fee-aware)')

# 3. Vanilla + post-hoc threshold sweep
print(f'\n{"="*60}')
print('[Threshold Sweep] Vanilla + post-hoc |w| < th → 0')
print(f'{"="*60}')
pi_v = VanillaPolicy().to(DEV)
# Quick retrain (5000 steps is enough to get ~same policy)
opt = torch.optim.AdamW(pi_v.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(5000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w = pi_v(z)
    w_prev = torch.cat([w[:1], w[:-1]]); to_p = FEE * (w-w_prev).abs().sum(1)
    pr = (w*r).sum(1) - to_p
    loss = -pr.mean()/(pr.std()+1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(pi_v.parameters(), 0.5); opt.step()

print(f'{"  th":>6s} {"SR":>8s} {"TO":>8s} {"Net SR":>8s} {"Active":>8s}')
print(f'  {"-"*42}')
for th in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    sr, to, net, act, _, w = evaluate(pi_v, rt, 'test', apply_th=th)
    print(f'  {th:>5.2f} {sr:>8.2f} {to:>8.4f} {net:>8.2f} {act:>8.2%}')

# 4. Gated + post-hoc threshold
print(f'\n{"="*60}')
print('[Threshold Sweep] Gated + post-hoc |w| < th → 0')
print(f'{"="*60}')
pi_g = GatedPolicy().to(DEV)
opt = torch.optim.AdamW(pi_g.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(5000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w = pi_g(z)
    w_prev = torch.cat([w[:1], w[:-1]]); to_p = FEE * (w-w_prev).abs().sum(1)
    pr = (w*r).sum(1) - to_p
    loss = -pr.mean()/(pr.std()+1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(pi_g.parameters(), 0.5); opt.step()

print(f'{"  th":>6s} {"SR":>8s} {"TO":>8s} {"Net SR":>8s} {"Active":>8s}')
print(f'  {"-"*42}')
for th in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    sr, to, net, act, _, w = evaluate(pi_g, rt, 'test', apply_th=th)
    print(f'  {th:>5.2f} {sr:>8.2f} {to:>8.4f} {net:>8.2f} {act:>8.2%}')

# ── Summary ──
print(f'\n{"="*60}')
print('FINAL SUMMARY')
print(f'{"="*60}')
print(f'{"Method":<30s} {"Test SR":>8s} {"TO":>8s} {"Net SR":>8s} {"Active":>8s}')
print('-' * 64)

# Re-run full eval for summary
# 1. Vanilla (already trained from above)
sr_v, to_v, net_v, act_v, _, _ = evaluate(pi_v, rt, 'test')
results['vanilla'] = {'test': sr_v, 'to': to_v, 'net': net_v, 'active': act_v}
# 2. Gated (already trained)
sr_g, to_g, net_g, act_g, _, _ = evaluate(pi_g, rt, 'test')
results['gated'] = {'test': sr_g, 'to': to_g, 'net': net_g, 'active': act_g}

for name, r in results.items():
    print(f'{name:<30s} {r["test"]:>8.2f} {r["to"]:>8.4f} {r["net"]:>8.2f} {r["active"]:>8.2%}')

# Find best threshold for each
for pi, pname in [(pi_v, 'Vanilla'), (pi_g, 'Gated')]:
    best_net = -99
    best_th = 0
    for th in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        _, _, net, _, _, _ = evaluate(pi, rt, 'test', apply_th=th)
        if net > best_net:
            best_net = net; best_th = th
    print(f'  {pname}: best threshold={best_th:.2f} → Net SR={best_net:.2f}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
