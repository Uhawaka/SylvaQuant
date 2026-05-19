#!/usr/bin/env python3 -u
"""
Dream World GRPO — train on CFM synthetic data, evaluate on real data.
Replicates stable train_rl_policy.py logic using experiment infrastructure.
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
from pipeline_cpcv import SYMBOLS, OUTPUT_DIR

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
ANNUAL = np.sqrt(24 * 365)
DZ, NC, H = 16, 9, 16  # H=16 from architecture search
LR = 3e-4
B = 4000
L = 5
K = 32
N_STEPS = 5000

# ── Policy (same as exp_c for fair comparison) ──
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu = nn.Linear(H, NC)
        self.log_std = nn.Parameter(torch.zeros(NC))
    def forward(self, s, det=False):
        h = self.net(s)
        m = torch.tanh(self.mu(h))
        if det: return m
        return torch.tanh(m + torch.randn_like(m) * self.log_std.exp())

# ── Load CFM synthetic data ──
print(f'═══ Dream World GRPO — CFM Synthetic → Real Data ═══')
print(f'Device: {DEV}')

d = np.load(ROOT / 'data/synthetic_cfm.npz')
lat_syn = d['latent'].astype(np.float32)  # (200K, 16)
ret_syn = d['returns'].astype(np.float32)  # (200K, 9)
print(f'Synthetic data: {len(lat_syn):,} pairs')

# ── Load real data (for evaluation only) ──
dr = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat_real = dr['latents'].astype(np.float32)
ret_real = dr['raw_ret'].astype(np.float32)
tr, va, te = dr['train_idx'], dr['val_idx'], dr['test_idx']

# Normalize real latents
rlm = lat_real[:tr[1]].mean(0, keepdims=True)
rls = lat_real[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_real_n = ((lat_real - rlm) / rls).astype(np.float32)

# Scale synthetic returns to match real vol
sc = ret_real.std() / (ret_syn.std() + 1e-8)
ret_syn *= sc
print(f'Return vol scaling: {sc:.4f}')
print(f'Synthetic ret std after scaling: {ret_syn.std():.6f}')
print(f'Real ret std: {ret_real.std():.6f}')

# ── Prepare synthetic segments ──
N_seg = len(lat_syn) // L
lat_syn = lat_syn[:N_seg * L].reshape(N_seg, L, DZ)
ret_syn = ret_syn[:N_seg * L].reshape(N_seg, L, NC)

# Normalize synthetic latents (within-segment)
lm = lat_syn.mean(axis=(0, 1), keepdims=True)
ls = lat_syn.std(axis=(0, 1), keepdims=True).clip(1e-6)
lat_syn_n = ((lat_syn - lm) / ls).astype(np.float32)

print(f'Segments: {N_seg} × L={L} = {N_seg * L} total steps')

S_tr = torch.from_numpy(lat_syn_n).to(DEV)
R_tr = torch.from_numpy(ret_syn).to(DEV)
Sr = torch.from_numpy(lat_real_n).to(DEV)  # real latents for eval
Rr = torch.from_numpy(ret_real).to(DEV)    # real returns for eval

# ── Evaluation on real data ──
@torch.no_grad()
def ev(p, split='test'):
    lo, hi = va if split == 'val' else te
    w = p(Sr[lo:hi], det=True)
    pr = (w * Rr[lo:hi]).sum(1).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    return sr, to

# ── GRPO Training ──
pi = Policy().to(DEV)
opt = torch.optim.AdamW(pi.parameters(), lr=LR)
print(f'\nTraining GRPO on synthetic data...')

best_sr = -10.0
t0 = time.time()

for step in range(N_STEPS):
    perm = torch.randperm(N_seg, device=DEV)[:B]
    s_seg = S_tr[perm]
    r_seg = R_tr[perm]

    total_loss = 0.0
    for l in range(L):
        s_t = s_seg[:, l]
        r_t = r_seg[:, l]

        # K samples per state
        sk = s_t.unsqueeze(1).expand(B, K, DZ).reshape(B * K, DZ)
        h = pi.net(sk)
        mu = torch.tanh(pi.mu(h))
        std = pi.log_std.exp().expand_as(mu)
        eps = torch.randn_like(mu)
        wk = torch.tanh(mu + eps * std)

        # Log prob (with tanh correction)
        lp = -0.5 * (eps**2 + 2 * pi.log_std + np.log(2 * np.pi))
        lp = lp - (2 * (np.log(2) - wk - F.softplus(-2 * wk)))
        lp = lp.sum(-1).view(B, K)
        wk = wk.view(B, K, NC)

        # Reward
        rk = r_t.unsqueeze(1).expand(-1, K, -1)
        rew = (wk * rk).sum(-1)

        # GRPO: group norm
        ad = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-8)

        # Loss
        pl = -(lp * ad.detach()).mean()
        el = -lp.mean()
        with torch.no_grad():
            mr = torch.tanh(pi.mu(pi.net(s_t)))
        kl = 0.5 * (mr**2 + pi.log_std.exp()**2 - 1 - 2 * pi.log_std).mean()
        total_loss += pl + 0.0005 * el + 0.01 * kl

    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    opt.step()

    if (step + 1) % 1000 == 0 or step == 0:
        sr_val, to_val = ev(pi, 'val')
        if sr_val > best_sr:
            best_sr = sr_val
        s = pi.log_std.exp().mean().item()
        print(f'  Step {step + 1:>5d}  Val SR={sr_val:.2f}  TO={to_val:.4f}  σ={s:.3f}')

# ── Final evaluation ──
sr_test, to_test = ev(pi, 'test')
sr_val_best = best_sr
print(f'\n[Dream GRPO] Time={time.time()-t0:.0f}s')
print(f'  Best Val: {sr_val_best:.2f}  Test SR: {sr_test:.2f}  Test TO: {to_test:.4f}')

# ── Compare with real-data trained ──
print(f'\n{"="*60}')
print('COMPARISON: Dream World vs Real-Data Training')
print(f'{"="*60}')
print(f'{"Method":<30s} {"Test SR":>8s} {"TO":>8s}')
print('-' * 48)
print(f'{"Dream GRPO (CFM synth)":<30s} {sr_test:>8.2f} {to_test:>8.4f}')
print(f'{"Real GRPO (H=16)":<30s} {"2.84":>8s} {"4.5185":>8s}')
print(f'{"Real Diff Sharpe (H=16)":<30s} {"2.64":>8s} {"0.3327":>8s}')

# Save
torch.save({'model_state': pi.state_dict(), 'n_coins': NC, 'latent_dim': DZ,
            'latent_mean': rlm, 'latent_std': rls},
           ROOT / 'data' / 'rl_policy_dream.pt')
print(f'\n✅ Saved: data/rl_policy_dream.pt')
