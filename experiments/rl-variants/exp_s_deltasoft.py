#!/usr/bin/env python3 -u
"""
DeltaSoft — 最优架构(θ=0.08)迭代: dropout / longer / L=16 / input noise.
"""
import sys, warnings, time, math
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
LR = 3e-4; B = 256; N_STEPS = 10000

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32); ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt = torch.from_numpy(lat_n).float()
rt = torch.from_numpy(ret).float()


class VecEnv:
    def __init__(self):
        self.lo, self.hi = tr; self.T = self.hi - self.lo
        self.lat, self.ret = lt, rt

    def roll(self, policy, L, noise=0.0):
        starts = torch.randint(0, self.T - L - 1, (B,))
        idx = starts.unsqueeze(1) + torch.arange(L).unsqueeze(0)
        zs = self.lat[idx]
        if noise > 0:
            zs = zs + torch.randn_like(zs) * noise
        rs = self.ret[idx]
        wp = torch.zeros(B, NC)
        cum = torch.zeros(B)
        for t in range(L):
            w = policy(zs[:, t], wp)
            cum += (w * rs[:, t]).sum(1) - FEE * (w - wp).abs().sum(1)
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


# ═══════════════════ VARIANTS ═══════════════════

def make_arch(variant):
    """Return a policy instance for the given variant name."""
    sd = DZ + NC

    if variant == 'baseline':
        # θ=0.08, L=32, no extra reg
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(sd, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
                self.sc = nn.Linear(H, NC); self.th = nn.Linear(H, NC)
            def forward(self, z, wp):
                h = self.enc(torch.cat([z, wp], dim=-1))
                d = torch.sign(self.sc(h)) * (self.sc(h).abs() - torch.sigmoid(self.th(h))*0.08).clamp(min=0)
                return (wp + d).clamp(-1, 1)
        return Net()

    elif variant == 'dropout':
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(sd, H), nn.SiLU(), nn.Dropout(0.1),
                    nn.Linear(H, H), nn.SiLU(), nn.Dropout(0.1))
                self.sc = nn.Linear(H, NC); self.th = nn.Linear(H, NC)
            def forward(self, z, wp):
                h = self.enc(torch.cat([z, wp], dim=-1))
                d = torch.sign(self.sc(h)) * (self.sc(h).abs() - torch.sigmoid(self.th(h))*0.08).clamp(min=0)
                return (wp + d).clamp(-1, 1)
        return Net()

    elif variant == 'longer':
        # baseline θ=0.08 but 20K steps (handled externally)
        return make_arch('baseline')

    elif variant == 'L16':
        # baseline θ=0.08 with L=16 (external param)
        return make_arch('baseline')

    elif variant == 'noise':
        # baseline θ=0.08 with input noise 0.01 (external param)
        return make_arch('baseline')

    elif variant == 'dropout_L16':
        # dropout + L=16
        net = make_arch('dropout')
        return net

    raise ValueError(variant)


configs = [
    # (variant, L, N_steps, noise, label)
    ('baseline', 32, 10000, 0.0, 'Baseline θ=0.08 L=32'),
    ('dropout', 32, 10000, 0.0, 'Dropout θ=0.08 L=32'),
    ('baseline', 16, 10000, 0.0, 'L=16 θ=0.08'),
    ('baseline', 32, 20000, 0.0, 'Longer 20K θ=0.08 L=32'),
    ('baseline', 32, 10000, 0.01, 'Noise(0.01) θ=0.08 L=32'),
    ('dropout', 16, 10000, 0.0, 'Dropout+L16 θ=0.08'),
]


results = []
for variant, L, n_steps, noise, label in configs:
    print(f'\n═══ {label} ═══', flush=True)
    pi = make_arch(variant)
    opt = torch.optim.AdamW(pi.parameters(), lr=LR, weight_decay=1e-5)
    env = VecEnv()

    gs, ns = evaluate(pi, va[0], va[1])
    print(f'  Init  Val GS={gs:.2f}  NS={ns:.2f}', flush=True)

    t0 = time.time()
    for step in range(n_steps):
        cum = env.roll(pi, L, noise)
        loss = -cum.mean() / max(cum.std(), 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()

        if (step + 1) % 5000 == 0 or step == 0:
            gs, ns = evaluate(pi, va[0], va[1])
            print(f'  Step {step+1:>5d}  Val GS={gs:.2f}  NS={ns:.2f}  wall={time.time()-t0:.0f}s', flush=True)

    gs, ns = evaluate(pi, te[0], te[1])
    print(f'  ═══ TEST ═══ GS={gs:.2f}  NS={ns:.2f}  wall={time.time()-t0:.0f}s', flush=True)
    results.append((label, gs, ns))


print('\n' + '═' * 60)
print('═══ FINAL SUMMARY ═══')
print(f'{"Variant":<30} {"Test GS":<10} {"Test NS":<10}')
print('─' * 52)
for label, gs, ns in results:
    print(f'{label:<30} {gs:<10.2f} {ns:<+10.2f}')
best = max(results, key=lambda x: x[2])
print(f'\n🏆 Best: {best[0]} — Test NS={best[2]:+.2f}')
