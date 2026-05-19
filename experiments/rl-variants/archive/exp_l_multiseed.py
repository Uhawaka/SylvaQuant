#!/usr/bin/env python3 -u
"""
Multi-seed validation of best DeltaSoft config (θ=0.15, L=32, B=256).
Runs 5 seeds → reports mean ± std.
"""
import sys, warnings, time, json
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn

NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004
DEV = 'cpu'

# ── Data ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).float()
rt = torch.from_numpy(ret).float()

# ── Policy ──
class DeltaSoftThresh(nn.Module):
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
        nn.init.constant_(self.thresh_head.bias, -2.0)
    def forward(self, z, w_prev):
        x = torch.cat([z, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)

# ── VecEnv ──
class VecEnv:
    def __init__(self, B=256, L=32, device='cpu'):
        self.B, self.L, self.device = B, L, device
        self.reset()
    def reset(self, split='train'):
        lo, hi = tr if split=='train' else (va if split=='val' else te)
        self.starts = np.random.randint(lo, hi - self.L - 1, size=self.B)
        self.w = torch.zeros(self.B, NC, device=self.device)
        self.t = 0
        self.split = split
    def _obs(self):
        ids = self.starts + self.t
        return lt[ids], self.w
    def step(self, w_new):
        ids = self.starts + self.t
        r = rt[ids]
        pr = (w_new * r).sum(1)
        to = (w_new - self.w).abs().sum(1)
        self.w = w_new; self.t += 1
        done = self.t >= self.L
        return pr - FEE * to, done
    def roll_out(self, pi):
        self.reset(self.split)
        rets = []
        for _ in range(self.L):
            z, wp = self._obs()
            wn = pi(z, wp)
            r, _ = self.step(wn)
            rets.append(r)
        return torch.stack(rets, dim=1)

# ── Eval ──
@torch.no_grad()
def eval_full(pi, split='test'):
    lo, hi = te if split=='test' else va
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
    fee = np.concatenate([[0.0], FEE * np.abs(np.diff(w_np, axis=0)).sum(1)])
    net = (pr - fee).mean() / max((pr - fee).std(), 1e-8) * ANNUAL
    act = (w_np.sum(1) > 0.001).mean()
    return sr, to, net, act

# ── Train ──
def train(pi, B=256, L=32, STEPS=10000, lr=3e-4, verbose=True):
    env = VecEnv(B=B, L=L)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    for step in range(STEPS):
        rets = env.roll_out(pi)
        pr_flat = rets.reshape(-1)
        loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        if verbose and (step % 2500 == 0 or step == 0):
            s, _, _, _ = eval_full(pi, 'val')
            print(f'  Step {step:>5d}  Val SR={s:.2f}  wall={time.time()-t0:.0f}s')
    if verbose:
        s, t, n, a = eval_full(pi, 'test')
        print(f'  Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:.2f}  Act={a:.2%}')
    return eval_full(pi, 'test')

# ══════════════════════════════════
# MULTI-SEED RUN
# ══════════════════════════════════
SEEDS = [42, 123, 456, 789, 1111]
CONFIGS = [
    {'name': 'DeltaSoft θ=0.15', 'theta_max': 0.15, 'L': 32, 'B': 256},
    {'name': 'DeltaSoft θ=0.20', 'theta_max': 0.20, 'L': 32, 'B': 256},
    {'name': 'DeltaSoft θ=0.30', 'theta_max': 0.30, 'L': 32, 'B': 256},
    # Try different L with best θ
    {'name': 'DeltaSoft L=16', 'theta_max': 0.15, 'L': 16, 'B': 256},
    {'name': 'DeltaSoft L=48', 'theta_max': 0.15, 'L': 48, 'B': 256},
]

for cfg in CONFIGS:
    print(f'\n{"="*60}')
    print(f'{cfg["name"]} — {len(SEEDS)} seeds')
    print(f'{"="*60}')
    outs = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        pi = DeltaSoftThresh(theta_max=cfg['theta_max'])
        s, to, n, a = train(pi, B=cfg['B'], L=cfg['L'], STEPS=10000, verbose=False)
        outs.append({'seed': seed, 'sr': s, 'to': to, 'net': n, 'act': a})
        print(f'  seed={seed:>4d}  SR={s:>7.2f}  TO={to:.4f}  Net SR={n:>+7.2f}  Act={a:.2%}')
    
    sr_v = [o['sr'] for o in outs]
    net_v = [o['net'] for o in outs]
    to_v = [o['to'] for o in outs]
    print(f'  ─────────────────────────────────────────────────────')
    print(f'  MEAN  SR={np.mean(sr_v):>7.2f}±{np.std(sr_v):.2f}  '
          f'Net SR={np.mean(net_v):>+7.2f}±{np.std(net_v):.2f}  '
          f'TO={np.mean(to_v):.4f}')
    print(f'  BEST  SR={max(sr_v):.2f}  WORST SR={min(sr_v):.2f}  '
          f'RANGE={max(sr_v)-min(sr_v):.2f}')
