#!/usr/bin/env python3 -u
"""
Diff Sharpe experiments — fee-aware, architecture search, sizing vs full weights.
All imports from stable src/ modules where possible.
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

# Use stable modules for data loading
from pipeline_cpcv import SYMBOLS

# ── Config ──
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NC = 9
DZ = 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004  # 0.04%

# ── Data (same source, clean alignment) ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
spnl = d['sig_pnl'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

# Normalize
lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)
st = torch.from_numpy(spnl).to(DEV)

# ── Policy architectures ──
class FullPolicy(nn.Module):
    """Full weight allocation: latent → [-1,1]^9 weights."""
    def __init__(self, H=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu = nn.Linear(H, NC)
    def forward(self, s):
        return torch.tanh(self.mu(self.net(s)))

class SizPolicy(nn.Module):
    """Single scalar per coin: latent → 1 sigmoid scalar × fixed equal weights."""
    def __init__(self, H=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1), nn.Sigmoid())
    def forward(self, s):
        k = self.net(s)  # (B, 1)
        return k * (1.0 / NC)  # scalar × equal weight

class SizPolicyPerCoin(nn.Module):
    """Per-coin sizing: latent → [0,∞)^9 scalars."""
    def __init__(self, H=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, NC), nn.Sigmoid())
    def forward(self, s):
        return self.net(s) * 2.0  # [0,2] per coin

# ── Evaluation ──
@torch.no_grad()
def evaluate(policy, ret_src, split='val'):
    lo, hi = va if split == 'val' else te
    w = policy(lt[lo:hi])
    pr = (w * ret_src[lo:hi]).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    return sr, to, pr

# ── Diff Sharpe with optional fee ──
def diff_sharpe(policy, ret_src, name, H=128, STEPS=20000, B=4096,
                lr=3e-4, with_fee=False, fee_scale=50.0, eval_every=1000):
    """Diff Sharpe optimization. If with_fee, adds turnover penalty."""
    print(f'\n{"="*60}')
    print(f'[Diff Sharpe] {name}  H={H}  fee={"Y" if with_fee else "N"}')
    print(f'{"="*60}')
    
    pi = policy.to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    best_sr = -10.0
    t0 = time.time()

    for step in range(STEPS):
        ts = np.random.randint(tr[0], tr[1] - B - 1)
        z = lt[ts:ts + B]
        r = rt[ts:ts + B] if ret_src == 'raw' else st[ts:ts + B]

        w = pi(z)

        if with_fee:
            # Portfolio return with turnover cost
            w_prev = torch.cat([w[:1], w[:-1]], dim=0)  # shift: w[t-1]
            to = (w - w_prev).abs().sum(1)  # turnover per bar
            pr = (w * r).sum(1) - FEE * to
        else:
            pr = (w * r).sum(1)

        mu = pr.mean()
        sd = pr.std() + 1e-8
        loss = -mu / sd

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        scheduler.step()

        if step % eval_every == 0 or step == 0:
            ret_src_t = rt if ret_src == 'raw' else st
            sr_val, to_val, _ = evaluate(pi, ret_src_t, 'val')
            if sr_val > best_sr:
                best_sr = sr_val
            if step % (eval_every * 2) == 0 or step == 0:
                print(f'  Step {step:>5d}  Val SR={sr_val:+.2f}  TO={to_val:.4f}')

    sr_test, to_test, pr_test = evaluate(pi, ret_src_t, 'test')
    print(f'  [{name}] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test SR={sr_test:.2f}  Test TO={to_test:.4f}')
    return {'val': best_sr, 'test': sr_test, 'to': to_test, 'pr': pr_test}

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

# 1. Arch search: H=16 (known stable)
results['full_H16_raw'] = diff_sharpe(FullPolicy(H=16), 'raw', 'full_H16_raw', H=16, B=8192)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 2. Arch search: H=32
results['full_H32_raw'] = diff_sharpe(FullPolicy(H=32), 'raw', 'full_H32_raw', H=32, B=8192)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 3. Arch search: H=64
results['full_H64_raw'] = diff_sharpe(FullPolicy(H=64), 'raw', 'full_H64_raw', H=64, B=8192)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 4. Fee-aware: H=16 with fee
results['full_H16_fee'] = diff_sharpe(FullPolicy(H=16), 'raw', 'full_H16_fee', H=16, B=8192, with_fee=True)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 5. Sizing per coin
results['siz_percoin_H16_raw'] = diff_sharpe(SizPolicyPerCoin(H=16), 'raw', 'siz_percoin_H16_raw', H=16, B=8192)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 6. Uniform sizing
results['siz_uniform_H16_raw'] = diff_sharpe(SizPolicy(H=16), 'raw', 'siz_uniform_H16_raw', H=16, B=8192)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# 7. Sig PnL with fee
results['full_H16_sig_fee'] = diff_sharpe(FullPolicy(H=16), 'sig', 'full_H16_sig_fee', H=16, B=8192, with_fee=True)
print(f'  ⏱ {time.time()-t_all:.0f}s')

# ── Summary ──
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'{"Experiment":<30s} {"Val SR":>8s} {"Test SR":>8s} {"TO":>8s}')
print('-' * 56)
for name, r in results.items():
    print(f'{name:<30s} {r["val"]:>8.2f} {r["test"]:>8.2f} {r["to"]:>8.4f}')
print(f'{"EW Baseline":<30s} {"—":>8s} {"-0.24":>8s} {"—":>8s}')
print(f'\nTotal: {time.time()-t_all:.0f}s')
