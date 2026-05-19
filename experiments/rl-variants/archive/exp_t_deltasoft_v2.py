#!/usr/bin/env python3 -u
"""
DeltaSoft v2 — 双路 encoder + per-coin return normalization.
Baseline 对比: 单一版本改进, 看净提升.

改进1 (Policy): 共享层→双路(ScoreNet + ThreshNet), 各自学习不同特征
改进2 (Env): per-coin return / σ, 均衡各币贡献
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import torch, torch.nn as nn, torch.nn.functional as F

SEED = 1111
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'cpu'
DZ, NC, H = 16, 9, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004
LR = 3e-4; B, L = 256, 32; N_STEPS = 10000
T_MAX = 0.08

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt = torch.from_numpy(lat_n).float()
rt = torch.from_numpy(ret).float()

# Per-coin sigma (training normalization)
coin_sigma = ret[:tr[1]].std(0, keepdims=True)  # (1, NC)
print(f'Coin sigma: {coin_sigma[0].round(4)}', flush=True)


# ═══════════════════ Policy Networks ═══════════════════

class DeltaSoftBaseline(nn.Module):
    """原版: 共享 encoder + 单层 score/thresh heads"""
    def __init__(self):
        super().__init__()
        sd = DZ + NC
        self.enc = nn.Sequential(nn.Linear(sd, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.sc = nn.Linear(H, NC)
        self.th = nn.Linear(H, NC)
    def forward(self, z, wp):
        h = self.enc(torch.cat([z, wp], dim=-1))
        d = torch.sign(self.sc(h)) * (self.sc(h).abs() - torch.sigmoid(self.th(h))*T_MAX).clamp(min=0)
        return (wp + d).clamp(-1, 1)


class DeltaSoftDualPath(nn.Module):
    """双路: 浅层共享 + 各自 MLP, score/thresh 用不同特征"""
    def __init__(self):
        super().__init__()
        sd = DZ + NC
        self.shared = nn.Linear(sd, H)  # 浅层: 只有1层
        self.score_net = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, NC))
        self.thresh_net = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, NC))
    def forward(self, z, wp):
        h = F.silu(self.shared(torch.cat([z, wp], dim=-1)))
        d = torch.sign(self.score_net(h)) * (self.score_net(h).abs() - torch.sigmoid(self.thresh_net(h))*T_MAX).clamp(min=0)
        return (wp + d).clamp(-1, 1)


# ═══════════════════ Environments ═══════════════════

class VecEnv:
    def __init__(self, normalize_returns=False):
        self.lo, self.hi = tr
        self.T = self.hi - self.lo
        self.lat, self.ret = lt, rt
        self.norm = normalize_returns
        self.cs = torch.from_numpy(coin_sigma).to(lt.device)

    def roll(self, policy, Lc):
        starts = torch.randint(0, self.T - Lc - 1, (B,))
        idx = starts.unsqueeze(1) + torch.arange(Lc).unsqueeze(0)
        zs = self.lat[idx]
        rs = self.ret[idx]
        if self.norm:
            rs = rs / self.cs  # per-coin normalization
        wp = torch.zeros(B, NC)
        cum = torch.zeros(B)
        for t in range(Lc):
            w = policy(zs[:, t], wp)
            pr = (w * rs[:, t]).sum(1)
            to = (w - wp).abs().sum(1)
            cum += pr - FEE * to
            wp = w.detach()
        return cum


@torch.no_grad()
def evaluate(policy, lo, hi):
    n = hi - lo
    w = torch.zeros(1, NC)
    gr, nr = [], []
    for i in range(n):
        wn = policy(lt[lo+i:lo+i+1], w)
        g = (wn * rt[lo+i:lo+i+1]).sum().item()
        nv = g - FEE * (wn - w).abs().sum().item()
        gr.append(g); nr.append(nv)
        w = wn
    gs = np.array(gr).mean() / max(np.array(gr).std(), 1e-8) * ANNUAL
    ns = np.array(nr).mean() / max(np.array(nr).std(), 1e-8) * ANNUAL
    return gs, ns


# ═══════════════════ Runner ═══════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run(policy_cls, env_normalize, label):
    pi = policy_cls()
    opt = torch.optim.AdamW(pi.parameters(), lr=LR, weight_decay=1e-5)
    env = VecEnv(normalize_returns=env_normalize)

    print(f'\n═══ {label} ({count_params(pi)} params) ═══', flush=True)
    gs, ns = evaluate(pi, va[0], va[1])
    print(f'  Init  Val GS={gs:.2f}  NS={ns:.2f}', flush=True)

    t0 = time.time()
    for step in range(N_STEPS):
        cum = env.roll(pi, L)
        loss = -cum.mean() / max(cum.std(), 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()

        if (step + 1) % 5000 == 0:
            gs, ns = evaluate(pi, va[0], va[1])
            print(f'  Step {step+1:>5d}  Val GS={gs:.2f}  NS={ns:.2f}  wall={time.time()-t0:.0f}s', flush=True)

    gs, ns = evaluate(pi, te[0], te[1])
    print(f'  ═══ TEST ═══ GS={gs:.2f}  NS={ns:.2f}  wall={time.time()-t0:.0f}s', flush=True)
    return gs, ns


results = []

# Baseline: 原版 + 原版 env
gs, ns = run(DeltaSoftBaseline, False, 'Baseline (shared enc + raw ret)')
results.append(('Baseline raw ret', gs, ns))

# Baseline + normalize
gs, ns = run(DeltaSoftBaseline, True, 'Baseline + norm ret')
results.append(('Baseline norm ret', gs, ns))

# Dual-path + raw
gs, ns = run(DeltaSoftDualPath, False, 'DualPath + raw ret')
results.append(('DualPath raw ret', gs, ns))

# Dual-path + normalize (完全体)
gs, ns = run(DeltaSoftDualPath, True, 'DualPath + norm ret')
results.append(('DualPath norm ret', gs, ns))

print('\n' + '═' * 60)
print('═══ SUMMARY ═══')
for label, gs, ns in results:
    marker = '🏆' if ns == max(r[2] for r in results) else ' '
    print(f'{marker} {label:<30} GS={gs:.2f}  NS={ns:+.2f}')
