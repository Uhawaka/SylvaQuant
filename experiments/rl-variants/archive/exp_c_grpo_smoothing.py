#!/usr/bin/env python3 -u
"""
Experiment C: GRPO vs Diff Sharpe + Post-hoc Smoothing Tradeoff.

Goals:
1. Fair comparison of GRPO and Diff Sharpe on same correct data
2. Post-hoc EMA smoothing to reduce turnover while preserving Sharpe
3. Find optimal smoothing level for fee-adjusted returns
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
from pipeline_cpcv import SYMBOLS

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NC, DZ = 9, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004

# ── Data (same as exp_b) ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)

# ── Policy (H=16 — known best capacity) ──
class Policy(nn.Module):
    def __init__(self, H=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu = nn.Linear(H, NC)
        self.log_std = nn.Parameter(torch.zeros(NC))  # for GRPO
    def forward(self, s, det=True):
        h = self.net(s)
        m = torch.tanh(self.mu(h))
        return m if det else torch.tanh(m + torch.randn_like(m) * self.log_std.exp())

# ── Eval (with optional post-hoc EMA) ──
def ema_smooth(w, alpha=0.5):
    """Apply EMA to weight sequence."""
    s = w.copy()
    for t in range(1, len(s)):
        s[t] = alpha * s[t] + (1 - alpha) * s[t-1]
    return s

@torch.no_grad()
def evaluate(policy, ret_src, split='val', ema_alpha=None):
    lo, hi = va if split == 'val' else te
    w = policy(lt[lo:hi], det=True).cpu().numpy()
    if ema_alpha is not None:
        w = ema_smooth(w, ema_alpha)
    pr = (w * ret[lo:hi]).sum(1)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w, axis=0)).sum(1).mean()
    # Fee-adjusted post-hoc (multi-coin: sum absolute changes across coins)
    dP = np.abs(np.diff(w, axis=0)).sum(1)  # (N-1,) — total turnover per bar
    dP = np.concatenate([[0.0], dP])        # pad first bar: no prior position
    fee_pr = pr - FEE * dP
    fee_sr = fee_pr.mean() / max(fee_pr.std(), 1e-8) * ANNUAL
    return sr, to, fee_sr, w

# ═══════════════════════════════════════════
# Part 1: GRPO on correct data
# ═══════════════════════════════════════════
def train_grpo():
    print(f'\n{"="*60}\n[GRPO] H=16, raw returns\n{"="*60}')
    
    # Segmented data
    L = 5; B = 4000; K = 32; STEPS = 10000
    
    n_seg = len(lt) // L
    S_tr = lt[:n_seg*L].reshape(n_seg, L, DZ)
    R_tr = rt[:n_seg*L].reshape(n_seg, L, NC)
    
    pi = Policy(H=16).to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=3e-4)
    best_sr = -10.0
    t0 = time.time()

    for step in range(STEPS):
        perm = torch.randperm(n_seg, device=DEV)[:B]
        ss = S_tr[perm]; rs = R_tr[perm]
        
        total_loss = 0.0
        for l in range(L):
            s_t = ss[:, l]; r_t = rs[:, l]
            sk = s_t.unsqueeze(1).expand(B, K, DZ).reshape(B*K, DZ)
            h = pi.net(sk); mu = torch.tanh(pi.mu(h))
            std = pi.log_std.exp().expand_as(mu)
            eps = torch.randn_like(mu); wk = torch.tanh(mu + eps * std)

            lp = -.5*(eps**2+2*pi.log_std+np.log(2*np.pi))
            lp = lp - (2*(np.log(2)-wk-F.softplus(-2*wk)))
            lp = lp.sum(-1).view(B, K); wk = wk.view(B, K, NC)

            rk = r_t.unsqueeze(1).expand(-1, K, -1)
            rew = (wk * rk).sum(-1)

            ad = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-8)
            pl = -(lp * ad.detach()).mean(); el = -lp.mean()
            with torch.no_grad(): mr = torch.tanh(pi.mu(pi.net(s_t)))
            kl = .5*(mr**2+pi.log_std.exp()**2-1-2*pi.log_std).mean()
            total_loss += pl + .0005*el + .01*kl

        opt.zero_grad(); total_loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step()

        if step % 2000 == 0 or step == 0:
            sr, to, fsr, _ = evaluate(pi, rt, 'val')
            if sr > best_sr: best_sr = sr
            print(f'  Step {step:>5d}  Val SR={sr:.2f}  TO={to:.4f}  σ={pi.log_std.exp().mean():.3f}')

    sr, to, fsr, w = evaluate(pi, rt, 'test')
    print(f'[GRPO] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test SR={sr:.2f}  TO={to:.4f}')
    return pi, {'val': best_sr, 'test': sr, 'to': to, 'fee_sr': fsr}

# ═══════════════════════════════════════════
# Part 2: Diff Sharpe (H=16) — same as exp_b best
# ═══════════════════════════════════════════
def train_diff_sharpe():
    print(f'\n{"="*60}\n[Diff Sharpe] H=16, raw returns\n{"="*60}')
    pi = Policy(H=16).to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=3e-4)
    best_sr, t0 = -10.0, time.time()

    for step in range(20000):
        ts = np.random.randint(tr[0], tr[1] - 8192 - 1)
        z, r = lt[ts:ts+8192], rt[ts:ts+8192]
        w = pi(z)
        pr = (w * r).sum(1)
        loss = -pr.mean() / (pr.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()

        if step % 4000 == 0 or step == 0:
            sr, to, fsr, _ = evaluate(pi, rt, 'val')
            if sr > best_sr: best_sr = sr
            print(f'  Step {step:>5d}  Val SR={sr:.2f}  TO={to:.4f}')

    sr, to, fsr, w = evaluate(pi, rt, 'test')
    print(f'[Diff Sharpe] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test SR={sr:.2f}  TO={to:.4f}')
    return pi, {'val': best_sr, 'test': sr, 'to': to, 'fee_sr': fsr}

# ═══════════════════════════════════════════
# Part 3: Post-hoc smoothing tradeoff
# ═══════════════════════════════════════════
def smoothing_tradeoff(policy, name):
    print(f'\n{"="*60}\n[Post-hoc Smoothing] {name}\n{"="*60}')
    print(f'{"alpha":>7s}  {"SR":>7s}  {"TO":>8s}  {"Fee SR":>8s}  {"Fee TO":>8s}  {"Σ|w|":>7s}')
    print('-' * 52)
    
    results = []
    for alpha in [None, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        sr, to, fsr, w = evaluate(policy, rt, 'test', ema_alpha=alpha)
        w_abs = np.abs(w).mean()
        alpha_str = f'{alpha}' if alpha is not None else 'raw'
        results.append((alpha_str, sr, to, fsr, w_abs))
        print(f'{alpha_str:>7s}  {sr:>7.2f}  {to:>8.4f}  {fsr:>8.2f}  {to*FEE*100:>8.2f}%  {w_abs:>7.3f}')
    
    return results

# ═══════════════════ RUN ═══════════════════
t_all = time.time()
results = {}

print(f'\n{"="*60}')
print('PART 1/3 — Training')
print(f'{"="*60}')

pi_diff, res_diff = train_diff_sharpe()
results['diff_sharpe'] = res_diff

pi_grpo, res_grpo = train_grpo()
results['grpo'] = res_grpo

print(f'\n{"="*60}')
print('PART 2/3 — Smoothing Tradeoff (Diff Sharpe)')
print(f'{"="*60}')
smooth_diff = smoothing_tradeoff(pi_diff, 'Diff Sharpe H=16')

print(f'\n{"="*60}')
print('PART 2/3 — Smoothing Tradeoff (GRPO)')
print(f'{"="*60}')
smooth_grpo = smoothing_tradeoff(pi_grpo, 'GRPO H=16')

# ── Summary ──
print(f'\n{"="*60}')
print('SUMMARY — GRPO vs Diff Sharpe')
print(f'{"="*60}')
print(f'{"Method":<20s} {"Val SR":>8s} {"Test SR":>8s} {"TO":>8s}')
print('-' * 46)
for name in ['diff_sharpe', 'grpo']:
    r = results[name]
    print(f'{name:<20s} {r["val"]:>8.2f} {r["test"]:>8.2f} {r["to"]:>8.4f}')

print(f'\n{"="*60}')
print('SUMMARY — Smoothing Tradeoff')
print(f'{"="*60}')
for name, sres in [('Diff Sharpe', smooth_diff), ('GRPO', smooth_grpo)]:
    print(f'\n--- {name} ---')
    print(f'{"alpha":>7s}  {"SR":>7s}  {"TO":>8s}  {"Fee SR":>8s}  {"Σ|w|":>7s}')
    for alpha_str, sr, to, fsr, wa in sres:
        print(f'{alpha_str:>7s}  {sr:>7.2f}  {to:>8.4f}  {fsr:>8.2f}  {wa:>7.3f}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
