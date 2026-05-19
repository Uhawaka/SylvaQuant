#!/usr/bin/env python3 -u
"""
DeltaGate v2 — w_prev outside policy (for MPS batching).
Policy outputs delta only. w_prev applied externally.
"""
import sys, warnings, time, math
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import torch, torch.nn as nn, torch.nn.functional as F
from pipeline_cpcv import SYMBOLS, OUTPUT_DIR

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DZ, NC, H = 16, 9, 64
LR = 3e-4; B = 4000; L = 5; K = 32; N_STEPS = 5000
FEE = 0.0004; T_MAX = 0.15

# ── Data ──
d = np.load('data/synthetic_cfm.npz')
lat = d['latent'].astype(np.float32); ret = d['returns'].astype(np.float32)
import pandas as pd
lr = np.load('data/market_latent.npy').astype(np.float32)
ld = np.load('data/market_latent_dates.npy', allow_pickle=True)
dm = {d: i for i, d in enumerate(ld)}
vm = np.zeros((len(lr), NC), np.float32)
for j, s in enumerate(SYMBOLS):
    v = np.load(OUTPUT_DIR / f'cpcv_vwap_{s}.npy')
    ds = pd.to_datetime(np.load(OUTPUT_DIR / f'cpcv_dates_{s}.npy'))
    for k, dt in enumerate(ds):
        i = dm.get(dt)
        if i is not None: vm[i, j] = v[k]
Nrr = len(lr) - 2
rr = np.zeros((Nrr, NC), np.float32)
for j in range(NC):
    r = vm[2:, j] / vm[1:-1, j] - 1.
    rr[:, j] = np.where(np.isfinite(r), r, 0.)
sc = rr.std() / (ret.std() + 1e-8)
ret *= sc

N_seg = len(lat) // L
lat = lat[:N_seg * L].reshape(N_seg, L, DZ)
ret = ret[:N_seg * L].reshape(N_seg, L, NC)
lm = lat.mean(axis=(0, 1), keepdims=True)
ls = lat.std(axis=(0, 1), keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

rla = lr[:Nrr]; rlm = rla.mean(0); rls = rla.std(0).clip(1e-6)
Sr = torch.from_numpy(((rla - rlm) / rls).astype(np.float32)).to(DEV)
Rr = torch.from_numpy(rr).to(DEV)
S_tr = torch.from_numpy(lat_n).to(DEV)
R_tr = torch.from_numpy(ret).to(DEV)
print(f'Data: {N_seg} segments × L={L}  |  Real market: {Nrr} bars', flush=True)


class DeltaGate(nn.Module):
    """Outputs delta = sign(score) × max(|score|-θ, 0). w = w_prev + delta (applied externally)."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.sc = nn.Linear(H, NC)
        self.th = nn.Linear(H, NC)
        self.log_std = nn.Parameter(torch.zeros(NC))

    def forward(self, s, det=False):
        h = self.shared(s)
        sc = self.sc(h); th = torch.sigmoid(self.th(h)) * T_MAX
        d = torch.sign(sc) * (sc.abs() - th).clamp(min=0)
        if det: return d
        std = self.log_std.exp().expand_as(sc)
        return torch.sign(sc + torch.randn_like(sc)*std) * ((sc + torch.randn_like(sc)*std).abs() - th).clamp(min=0)

    def sample_k(self, s_t, Kc):
        Bc = s_t.shape[0]
        sk = s_t.unsqueeze(1).expand(Bc, Kc, DZ).reshape(Bc * Kc, DZ)
        h = self.shared(sk)
        sc = self.sc(h); th = torch.sigmoid(self.th(h)) * T_MAX
        std = self.log_std.exp().expand_as(sc); eps = torch.randn_like(sc)
        dn = torch.sign(sc + eps*std) * ((sc + eps*std).abs() - th).clamp(min=0)
        lp = -0.5*(eps**2 + 2*self.log_std + math.log(2*math.pi))
        lp = lp - (2*(math.log(2)-(sc+eps*std)-F.softplus(-2*(sc+eps*std))))
        return dn.view(Bc, Kc, NC), lp.sum(-1).view(Bc, Kc)


@torch.no_grad()
def ev(p):
    """Eval on real market. w = clamp(w_prev + delta, -1, 1) tracked externally."""
    w = torch.zeros(1, NC, device=DEV)
    ws = []
    for i in range(Nrr):
        d = p.forward(Sr[i:i+1], det=True)
        w = (w + d).clamp(-1, 1)
        ws.append(w.cpu().numpy().ravel())
    ws = np.array(ws)
    pr = (torch.from_numpy(ws).to(DEV) * Rr).sum(1).cpu().numpy()
    to_coin = np.abs(np.diff(ws, axis=0)).sum(1).mean()
    fee_bar = np.concatenate([[FEE*ws[0].sum()], FEE*np.abs(np.diff(ws, axis=0)).sum(1)])
    gsr = pr.mean()/max(pr.std(),1e-8)*np.sqrt(252*96)
    nsr = (pr-fee_bar).mean()/max((pr-fee_bar).std(),1e-8)*np.sqrt(252*96)
    return gsr, nsr, to_coin, np.abs(ws).mean()


pi = DeltaGate().to(DEV)
opt = torch.optim.AdamW(pi.parameters(), lr=LR)
print(f'═══ DeltaGate v2 (B={B} L={L} K={K} θ={T_MAX}) ═══')
print(f'Params: {sum(p.numel() for p in pi.parameters())}', flush=True)

gsr, nsr, to0, sw0 = ev(pi)
print(f'Init  GS={gsr:.2f}  NS={nsr:.2f}  Σ|w|={sw0:.2f}  TO={to0:.4f}', flush=True)

t0 = time.time()
for step in range(N_STEPS):
    perm = torch.randperm(N_seg, device=DEV)[:B]
    s_seg = S_tr[perm]; r_seg = R_tr[perm]
    total_loss = 0
    w_prev = torch.zeros(B, NC, device=DEV)

    for l in range(L):
        s_t = s_seg[:, l]; r_t = r_seg[:, l]
        dk, lp = pi.sample_k(s_t, K)  # (B, K, NC) deltas
        wk = (w_prev.unsqueeze(1) + dk).clamp(-1, 1)

        fee = FEE * wk.abs().sum(-1) if l == 0 else FEE * (wk - w_prev.unsqueeze(1)).abs().sum(-1)
        rk = r_t.unsqueeze(1).expand(-1, K, -1)
        rew = (wk * rk).sum(-1) - fee
        ad = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-8)

        pl = -(lp * ad.detach()).mean()
        el = -lp.mean()
        with torch.no_grad(): mr = pi.sc(pi.shared(s_t))
        kl = 0.5 * (mr**2 + pi.log_std.exp()**2 - 1 - 2 * pi.log_std).mean()
        total_loss += pl + 0.0005 * el + 0.01 * kl

        with torch.no_grad():
            w_prev = (w_prev + pi.forward(s_t, det=True)).clamp(-1, 1)

    opt.zero_grad(); total_loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step()

    if (step+1) % 1000 == 0 or step == 0:
        gsr, nsr, to_ev, sw = ev(pi)
        print(f'  Step {step+1:>5d}  GS={gsr:.2f}  NS={nsr:.2f}  Σ|w|={sw:.2f}  TO={to_ev:.4f}  wall={time.time()-t0:.0f}s', flush=True)

gsr, nsr, to_ev, sw = ev(pi)
print(f'\n═══ FINAL ═══\nGS={gsr:.2f}  NS={nsr:.2f}  Σ|w|={sw:.2f}  TO={to_ev:.4f}', flush=True)
print(f'Time: {time.time()-t0:.0f}s', flush=True)
torch.save({'model_state': pi.state_dict(), 'fee': FEE, 'to': to_ev}, 'data/rl_policy.pt')
