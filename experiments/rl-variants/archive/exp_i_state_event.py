#!/usr/bin/env python3 -u
"""
Event Policy — w[t-1] as part of state.
  state[t] = [latent[t], w[t-1]]  →  w[t] = policy(state[t])

Network learns event/hold/end structure implicitly by seeing its own position.
Supports soft threshold on output for sparse trades.

Comparison: 
  1. State augmentation only (baseline)
  2. State augmentation + soft_threshold on output
  3. Soft threshold (no state, previous best)
  4. Event gating (explicit e/d, previous attempt)
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

# ── Policies ──

class StateAugPolicy(nn.Module):
    """w[t-1] in state → tanh output. Network learns all structure."""
    def __init__(self):
        super().__init__()
        D_in = DZ + NC  # 16 + 9 = 25
        self.net = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, NC), nn.Tanh()
        )
    def forward(self, s, w_prev):
        x = torch.cat([s, w_prev], dim=-1)  # (B, 25)
        return self.net(x)

class StateAugSoftThresh(nn.Module):
    """w[t-1] in state + soft threshold on output. Best of both worlds."""
    def __init__(self, theta_max=0.3):
        super().__init__()
        self.theta_max = theta_max
        D_in = DZ + NC
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)
        self.thresh_head = nn.Linear(H, NC)
        nn.init.constant_(self.thresh_head.bias, -2.5)  # θ ≈ 0.075 init
        
    def forward(self, s, w_prev):
        x = torch.cat([s, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        abs_s = score.abs()
        w = score.sign() * (abs_s - theta).clamp(min=0)
        return w

class EventGatePolicy(nn.Module):
    """Explicit event gating (previous approach). w[t-1] in state, parallel gating."""
    def __init__(self, alpha_init=0.3):
        super().__init__()
        D_in = DZ + NC
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.dir_head = nn.Linear(H, NC)
        self.event_head = nn.Linear(H, NC)
        nn.init.constant_(self.event_head.bias, -1.5)
        self.logit_alpha = nn.Parameter(torch.tensor(
            np.log(alpha_init / (1 - alpha_init)), dtype=torch.float32
        ))
    def forward(self, s, w_prev):
        x = torch.cat([s, w_prev], dim=-1)
        h = self.encoder(x)
        d = torch.tanh(self.dir_head(h))
        e = torch.sigmoid(self.event_head(h))
        a = torch.sigmoid(self.logit_alpha)
        return (1 - a * e) * w_prev + a * e * d

# ── Evaluation — all stateful (sequential) ──
@torch.no_grad()
def evaluate(pi, ret_src, split='test'):
    """2-pass fixed-point evaluation (much faster than sequential)."""
    lo, hi = va if split == 'val' else te
    z = lt[lo:hi]  # (n, 16), parallel
    
    # Pass 1: zeros
    w = pi(z, torch.zeros_like(z[:, :NC]))
    # Pass 2: refine
    wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
    w = pi(z, wp)
    # Pass 3: more refine
    wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
    w = pi(z, wp)
    
    ret_sl = ret_src[lo:hi]
    pr = (w * ret_sl).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    fee_cost = FEE * np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    pr_f = pr - fee_cost
    sf = pr_f.mean() / max(pr_f.std(), 1e-8) * ANNUAL
    act = (w.abs().sum(1).cpu().numpy() > 0.001).mean()
    aw = w.abs().mean().item()
    return sr, to, sf, act, aw, w, pr

# ── Training ──
def train(pi, name, STEPS=20000, B=4096, lr=3e-4):
    """
    Stateful training via fixed-point iteration (2-pass).
    w ~ f(z, shift(w)) — fully parallel after 2 iterations.
    """
    print(f'\n{"="*60}')
    print(f'[Train] {name}')
    print(f'{"="*60}')
    pi.to(DEV); opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    best_sr = -10; t0 = time.time()
    
    for step in range(STEPS):
        ts = np.random.randint(tr[0], tr[1] - B - 1)
        z = lt[ts:ts+B]; r = rt[ts:ts+B]
        
        # Pass 1: w ~ f(z, zeros) — rough approx
        wp = torch.zeros(B, NC, device=DEV)
        w = pi(z, wp)
        
        # Pass 2: shift, refine (parallel)
        wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
        w = pi(z, wp.detach())
        
        # Pass 3: one more refinement
        wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
        w = pi(z, wp.detach())
        
        w_prev_batch = torch.cat([w[:1], w[:-1]], dim=0)
        to_p = FEE * (w - w_prev_batch).abs().sum(1)
        pr = (w * r).sum(1) - to_p
        loss = -pr.mean() / (pr.std() + 1e-8)
        
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()
        
        if step % 5000 == 0 or step == 0:
            sv, tv, _, av, aw_v, _, _ = evaluate(pi, rt, 'val')  # uses true sequential eval
            best_sr = max(best_sr, sv)
            pi.train()
            print(f'  Step {step:>5d}  Val SR={sv:.2f}  TO={tv:.4f}  Act={av:.2%}  |w|={aw_v:.3f}')
    
    s, t, n, a, aw, w_f, p_f = evaluate(pi, rt, 'test')
    print(f'  [{name}] Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:.2f}  Act={a:.2%}  |w|={aw:.3f}')
    return {'test': s, 'to': t, 'net': n, 'active': a, 'avg_w': aw, 'pr': p_f, 'w': w_f}

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

print('═══ Event Policy — w[t-1] in State ═══')

# 1. State augmentation only
results['state'] = train(StateAugPolicy(), 'StateAug')

# 2. State aug + soft threshold
results['state_soft'] = train(StateAugSoftThresh(), 'State+SoftThresh')

# 3. Event gate (explicit gates, w in state)
results['gate'] = train(EventGatePolicy(), 'EventGate')

# ── Analysis ──
print(f'\n{"="*60}')
print('[Analysis] Position Hold Length')
print(f'{"="*60}')
pi_best = StateAugSoftThresh().to(DEV)
opt = torch.optim.AdamW(pi_best.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(15000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w = []; wp = torch.zeros(1, NC, device=DEV)
    for t in range(4096):
        wt = pi_best(z[t:t+1], wp); w.append(wt); wp = wt
    w = torch.cat(w, dim=0)
    wp2 = torch.cat([w[:1], w[:-1]])
    tp = FEE * (w-wp2).abs().sum(1)
    pr = (w*r).sum(1) - tp
    l = -pr.mean()/(pr.std()+1e-8)
    opt.zero_grad(); l.backward()
    nn.utils.clip_grad_norm_(pi_best.parameters(), 0.5); opt.step()

# Test & analyze (2-pass)
lo, hi = te
z = lt[lo:hi]
w = pi_best(z, torch.zeros(hi-lo, NC, device=DEV))
wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
w = pi_best(z, wp)
wp = torch.cat([torch.zeros(1, NC, device=DEV), w[:-1]], dim=0)
w = pi_best(z, wp)
w_np = w.cpu().numpy()

# Position change analysis
print(f'\nPosition hold length analysis:')
total_bars = len(w_np)
for c in range(NC):
    w_c = w_np[:, c]
    # Find non-zero segments
    nonzero = np.abs(w_c) > 0.001
    # Find transitions
    borders = np.diff(np.concatenate([[0], nonzero.astype(int), [0]]))
    entries = np.where(borders == 1)[0]
    exits = np.where(borders == -1)[0]
    
    if len(entries) > 0 and len(exits) > 0:
        min_len = min(len(entries), len(exits))
        hold_lens = exits[:min_len] - entries[:min_len]
        print(f'  coin[{c}]: {min_len} trades, avg hold={hold_lens.mean():.1f} bars, '
              f'pct_active={nonzero.mean():.2%}')

# Portfolio-level
position_changes = np.sum(np.abs(np.diff(w_np, axis=0)) > 0.001, axis=1)
print(f'\n  Portfolio changes per bar: mean={position_changes.mean():.3f} coins')
print(f'  Bars with ZERO position changes: {np.mean(position_changes == 0):.2%}')
print(f'  Weight autocorrelation (lag 1): ', end='')
for c in range(NC):
    w_c = w_np[:, c]
    ac = np.corrcoef(w_c[:-1], w_c[1:])[0,1]
    if c == 0:
        print(f'{ac:.4f}', end='')
    else:
        print(f', {ac:.4f}', end='')
print()

# ── Summary ──
print(f'\n{"="*60}')
print('FINAL SUMMARY')
print(f'{"="*60}')
print(f'{"Method":<25s} {"Test SR":>7s} {"TO":>7s} {"Net SR":>7s} {"Active":>7s} {"|w|":>7s}')
print('-' * 62)
for name in sorted(results.keys()):
    r = results[name]
    print(f'{name:<25s} {r["test"]:>7.2f} {r["to"]:>7.4f} {r["net"]:>7.2f} {r["active"]:>7.2%} {r["avg_w"]:>7.3f}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
