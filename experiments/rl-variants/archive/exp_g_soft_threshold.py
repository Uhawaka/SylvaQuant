#!/usr/bin/env python3 -u
"""
Soft Threshold Policy — Diff Sharpe with built-in sparse weights.
  w = sign(s) · max(|s| - θ, 0)
where θ is input-dependent: larger when uncertain → fewer trades.
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
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)

# ── Soft Threshold Policy ──
class SoftThresholdPolicy(nn.Module):
    """
    w = sign(s) · max(|s| - θ, 0)
    where s = score(s) ∈ ℝ^9, θ ∈ [0, θ_max]^9
    Fully differentiable (subgradient at |s|=θ).
    """
    def __init__(self, theta_max=0.3):
        super().__init__()
        self.theta_max = theta_max
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.score_head = nn.Linear(H, NC)
        self.theta_head = nn.Linear(H, NC)
        
        # Init: theta ≈ 0.05 initially (allow small trades, learn to shut off)
        nn.init.constant_(self.theta_head.bias, -2.5)  # sigmoid(-2.5) ≈ 0.075
        
    def forward(self, s):
        h = self.net(s)
        score = self.score_head(h)  # unbounded
        theta = torch.sigmoid(self.theta_head(h)) * self.theta_max  # [0, θ_max]
        
        # Soft threshold (subgradient at |s|=θ → 0)
        abs_score = score.abs()
        sign = score.sign()
        w = sign * (abs_score - theta).clamp(min=0.0)
        
        # Track gate ratio for analysis
        active = (abs_score > theta).float().mean()
        
        return w, {'theta': theta, 'active_ratio': active, 'abs_score': abs_score}

class HystPolicy(nn.Module):
    """
    Soft threshold WITH hysteresis:
    w_t = sign(s) · max(|s| - θ_on, 0)
          + sign(w_{t-1}) · min(|w_{t-1}|, θ_off)
    
    Entry requires |s| > θ_on (higher bar)
    Exit requires w stays until the position would be closed naturally
    
    Simplified: use different thresholds for entry and exit
    θ_entry = θ_base + θ_delta (harder to enter)
    θ_exit  = θ_base - θ_delta (harder to exit, i.e., hold position longer)
    
    But for vectorized batch training (no sequential), we need:
    w_t = sign(s) · max(|s| - θ_t, 0) where θ_t depends on w_prev
    
    Simplified hysteresis:
    θ_eff = θ_base - α · |w_prev|
    → if you're in a position, threshold is lower (harder to exit)
    → if you're flat, threshold is higher (harder to enter)
    """
    def __init__(self, theta_max=0.3, hyst_alpha=0.5):
        super().__init__()
        self.theta_max = theta_max
        self.hyst_alpha = hyst_alpha
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.score_head = nn.Linear(H, NC)
        self.theta_head = nn.Linear(H, NC)
        nn.init.constant_(self.theta_head.bias, -2.5)
        
    def forward(self, s, w_prev=None):
        h = self.net(s)
        score = self.score_head(h)
        theta_base = torch.sigmoid(self.theta_head(h)) * self.theta_max
        
        if w_prev is not None:
            # Hysteresis: reduce threshold when already in position
            theta_eff = theta_base - self.hyst_alpha * w_prev.abs()
            theta_eff = theta_eff.clamp(min=0.01)
        else:
            theta_eff = theta_base
        
        abs_score = score.abs()
        sign = score.sign()
        w = sign * (abs_score - theta_eff).clamp(min=0.0)
        
        active = (abs_score > theta_eff).float().mean()
        return w, {'theta': theta_eff, 'active_ratio': active}

# ── Evaluation ──
@torch.no_grad()
def evaluate(pi, ret_src, split='test', hyst=False):
    lo, hi = va if split == 'val' else te
    if hyst:
        # Sequential evaluation for hysteresis
        ws = []
        w_prev = None
        for i in range(lo, hi):
            s_t = lt[i:i+1]
            w_t, _ = pi(s_t, w_prev)
            ws.append(w_t)
            w_prev = w_t
        w = torch.cat(ws, dim=0)
    else:
        w, _ = pi(lt[lo:hi])
    
    pr = (w * ret_src[lo:hi]).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    
    fee_cost = FEE * np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    pr_fee = pr - fee_cost
    sr_fee = pr_fee.mean() / max(pr_fee.std(), 1e-8) * ANNUAL
    
    active_ratio = (w.abs().sum(1).cpu().numpy() > 0.001).mean()
    avg_w = w.abs().mean().item()
    
    return sr, to, sr_fee, active_ratio, avg_w, pr, w

# ── Diff Sharpe Training ──
def train(pi, ret_src, name, STEPS=20000, B=4096, lr=3e-4, hyst=False):
    print(f'\n{"="*60}')
    print(f'[Train] {name}')
    print(f'{"="*60}')
    
    pi.to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    best = {'sr': -10}
    t0 = time.time()
    
    for step in range(STEPS):
        ts = np.random.randint(tr[0], tr[1] - B - 1)
        z = lt[ts:ts+B]
        r = rt[ts:ts+B] if ret_src == 'raw' else st[ts:ts+B]
        
        if hyst:
            # Sequential within each batch (slow but necessary for hysteresis)
            ws = []
            w_prev = None
            for i in range(B):
                w_t, _ = pi(z[i:i+1], w_prev)
                ws.append(w_t)
                w_prev = w_t
            w = torch.cat(ws, dim=0)
        else:
            w, info = pi(z)
        
        w_prev_batch = torch.cat([w[:1], w[:-1]], dim=0)
        to_pen = FEE * (w - w_prev_batch).abs().sum(1)
        pr = (w * r).sum(1) - to_pen
        
        loss = -pr.mean() / (pr.std() + 1e-8)
        
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()
        
        if step % 5000 == 0 or step == 0:
            sr_v, to_v, _, act_v, aw_v, _, _ = evaluate(pi, rt, 'val', hyst=hyst)
            pi.train()
            if sr_v > best['sr']:
                best['sr'] = sr_v
            theta_info = info.get('theta', torch.zeros(1))
            print(f'  Step {step:>5d}  Val SR={sr_v:.2f}  TO={to_v:.4f}  '
                  f'Act={act_v:.2%}  Avg|w|={aw_v:.3f}  θ≈{theta_info.mean().item():.4f}')
    
    sr_t, to_t, net_t, act_t, aw_t, pr_t, w_t = evaluate(pi, rt, 'test', hyst=hyst)
    print(f'  [{name}] Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={sr_t:.2f}  TO={to_t:.4f}  Net SR={net_t:.2f}  Act={act_t:.2%}  Avg|w|={aw_t:.3f}')
    
    return {'test': sr_t, 'to': to_t, 'net': net_t, 'active': act_t, 'avg_w': aw_t, 'pr': pr_t, 'w': w_t}

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

# 1. Soft Threshold (no fee in loss)
results['softthresh'] = train(SoftThresholdPolicy(theta_max=0.3), 'raw', 'SoftThreshold θ_max=0.3')

# 2. Soft Threshold (lower max)
results['softthresh_02'] = train(SoftThresholdPolicy(theta_max=0.2), 'raw', 'SoftThreshold θ_max=0.2')

# 3. Soft Threshold (higher max)
results['softthresh_05'] = train(SoftThresholdPolicy(theta_max=0.5), 'raw', 'SoftThreshold θ_max=0.5')

# 4. Hysteresis (takes longer — sequential eval)
# results['hyst'] = train(HystPolicy(theta_max=0.3), 'raw', 'Hysteresis θ_max=0.3', hyst=True)

# ── Analysis: threshold behavior ──
print(f'\n{"="*60}')
print('[Analysis] Learned Threshold Distribution')
print(f'{"="*60}')

# Load the best soft threshold policy
pi_best = SoftThresholdPolicy(theta_max=0.3).to(DEV)
# Quick retrain
opt = torch.optim.AdamW(pi_best.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(10000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w, info = pi_best(z)
    w_prev = torch.cat([w[:1], w[:-1]]); to_p = FEE * (w-w_prev).abs().sum(1)
    pr = (w*r).sum(1) - to_p
    loss = -pr.mean()/(pr.std()+1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(pi_best.parameters(), 0.5); opt.step()

# Analyze on test set
with torch.no_grad():
    w_test, info_test = pi_best(lt[te[0]:te[1]])
    theta = info_test['theta'].cpu().numpy()
    abs_score = info_test['abs_score'].cpu().numpy()
    active = info_test['active_ratio'].item()

print(f'  Active ratio: {active:.2%}')
print(f'  Theta stats per coin:')
for c in range(NC):
    t_c = theta[:, c]
    print(f'    coin[{c}]: mean={t_c.mean():.4f}  std={t_c.std():.4f}  '
          f'min={t_c.min():.4f}  max={t_c.max():.4f}')

# Compare with vanilla + post-hoc
print(f'\n  # Bars with w=0 (all coins): {np.sum(w_test.abs().sum(1).cpu().numpy() < 0.001):,}/{w_test.shape[0]:,}')

# ── Summary ──
print(f'\n{"="*60}')
print('FINAL SUMMARY')
print(f'{"="*60}')
print(f'{"Method":<30s} {"Test SR":>8s} {"TO":>8s} {"Net SR":>8s} {"Active":>8s} {"|w|":>8s}')
print('-' * 72)
for name in sorted(results.keys()):
    r = results[name]
    print(f'{name:<30s} {r["test"]:>8.2f} {r["to"]:>8.4f} {r["net"]:>8.2f} {r["active"]:>8.2%} {r["avg_w"]:>8.3f}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
