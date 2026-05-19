#!/usr/bin/env python3 -u
"""
GRPO + Gate — policy outputs direction (tanh) × gate (sigmoid).
Gate initialized closed (bias=-5), learns to open only when confident.
Entry cost on first step: FEE × |w| = FEE × gate × |direction|
  → if gate≈0, no cost, no trade.
  → policy learns: only open gate when expected return > entry cost.

Architecture: direction_head ∈ [-1,1]^(NC) and gate_head ∈ [0,1]^(NC)
"""
import sys, warnings, time
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import torch, torch.nn as nn, torch.nn.functional as F
from pipeline_cpcv import SYMBOLS, OUTPUT_DIR

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DZ, NC, H = 16, 9, 128
LR = 3e-4; B = 4000; L = 5; K = 32; N_STEPS = 5000; FEE = 0.0004


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(DZ, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.mu_head = nn.Linear(H, NC)                     # direction
        self.gate_head = nn.Linear(H, NC)                   # per-coin confidence
        self.log_std = nn.Parameter(torch.zeros(NC))
        # Initialize gate to "closed": bias = -5 → sigmoid ≈ 0.007
        nn.init.constant_(self.gate_head.bias, -5.0)

    def forward(self, s, det=False):
        h = self.shared(s)
        mu = torch.tanh(self.mu_head(h))
        gate = torch.sigmoid(self.gate_head(h))
        w = gate * mu
        if det:
            return w
        # Exploration: add noise to pre-squashed values, then re-squash
        mu_noisy = mu + torch.randn_like(mu) * self.log_std.exp()
        # Recompute noisy weight = gate * noisy_mu
        w_noisy = gate * torch.tanh(mu_noisy)  # keep direction noise, gate deterministic
        return w_noisy

    def sample_k(self, s_t, Kc):
        """K samples per state. Returns (B, K, NC) weights and (B, K) log_probs."""
        Bc = s_t.shape[0]
        sk = s_t.unsqueeze(1).expand(Bc, Kc, DZ).reshape(Bc * Kc, DZ)
        h = self.shared(sk)
        mu = torch.tanh(self.mu_head(h))
        gate = torch.sigmoid(self.gate_head(h))

        std = self.log_std.exp().expand_as(mu)
        eps = torch.randn_like(mu)
        mu_noisy = torch.tanh(mu + eps * std)
        wk = gate * mu_noisy  # (B*K, NC)

        # Log prob of tanh-squashed Gaussian (direction)
        lp = -0.5 * (eps**2 + 2 * self.log_std + np.log(2 * np.pi))
        # Tanh correction
        pre_tanh = mu + eps * std
        lp = lp - (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh)))
        lp = lp.sum(-1).view(Bc, Kc)
        return wk.view(Bc, Kc, NC), lp


# ── Load & prepare data (identical to original) ──
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
        if i is not None:
            vm[i, j] = v[k]
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

rla = lr[:Nrr]
rlm = rla.mean(0)
rls = rla.std(0).clip(1e-6)
Sr = torch.from_numpy(((rla - rlm) / rls).astype(np.float32)).to(DEV)
Rr = torch.from_numpy(rr).to(DEV)
S_tr = torch.from_numpy(lat_n).to(DEV)
R_tr = torch.from_numpy(ret).to(DEV)

print(f'Segments: {N_seg} × L={L} = {N_seg * L} total steps', flush=True)


@torch.no_grad()
def ev(p):
    """Eval on real market data. Returns: gross_sr, gross_pnl, w_mean, sum_abs_w, to"""
    w = p.forward(Sr, det=True)
    pr = (w * Rr).sum(1)
    sr = pr.mean() / (pr.std() + 1e-8) * np.sqrt(252 * 96)
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    # Net SR (post-fee)
    fee_per_bar = FEE * np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1)
    fee_per_bar = np.concatenate([[FEE * w[0].abs().sum().item()], fee_per_bar])  # entry on first bar
    nr = pr.cpu().numpy() - fee_per_bar
    nsr = nr.mean() / max(nr.std(), 1e-8) * np.sqrt(252 * 96)
    return (sr.item(), pr.mean().item(), w.mean(0).cpu().numpy(),
            w.abs().sum(1).cpu().mean().item(), to, nsr.item())


pi = Policy().to(DEV)
opt = torch.optim.AdamW(pi.parameters(), lr=LR)
print(f'═══ GRPO + Gate (B={B}, L={L}, K={K}, FEE={FEE}) ═══', flush=True)

# Initial eval with gate closed
sr0, pn0, wm0, sw0, to0, nsr0 = ev(pi)
print(f'Init  Gross SR={sr0:.2f}  Net SR={nsr0:.2f}  Σ|w|={sw0:.2f}  TO={to0:.4f}  '
      f'gate=[{torch.sigmoid(pi.gate_head.bias).mean().item():.4f}]', flush=True)

t0 = time.time()
for step in range(N_STEPS):
    perm = torch.randperm(N_seg, device=DEV)[:B]
    s_seg = S_tr[perm]; r_seg = R_tr[perm]  # (B, L, DZ), (B, L, NC)

    total_loss = 0
    w_prev = None
    for l in range(L):
        s_t = s_seg[:, l]     # (B, DZ)
        r_t = r_seg[:, l]     # (B, NC)

        # K samples per state
        wk, lp = pi.sample_k(s_t, K)  # (B, K, NC), (B, K)

        # Entry / turnover cost (leverage gate = closed → no cost)
        if l == 0:
            fee = FEE * wk.abs().sum(-1)           # entry: FEE × |w|
        else:
            fee = FEE * (wk - w_prev).abs().sum(-1) # turnover: FEE × |Δw|
        w_prev = wk.detach()

        # Reward = portfolio return - fee
        rk = r_t.unsqueeze(1).expand(-1, K, -1)
        rew = (wk * rk).sum(-1) - fee

        # GRPO: group norm within each (segment, step)
        ad = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-8)

        # Loss: PG + entropy + KL
        pl = -(lp * ad.detach()).mean()
        el = -lp.mean()
        with torch.no_grad():
            h_ref = pi.shared(s_t)
            mr = torch.tanh(pi.mu_head(h_ref))
        # KL is on direction only (gate is deterministic)
        kl = 0.5 * (mr**2 + pi.log_std.exp()**2 - 1 - 2 * pi.log_std).mean()
        total_loss += pl + 0.0005 * el + 0.01 * kl

    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    opt.step()

    if (step + 1) % 1000 == 0 or step == 0:
        sr, pn, wm, sw, to, nsr = ev(pi)
        s = pi.log_std.exp().mean().item()
        # Check gate stats
        with torch.no_grad():
            h_eval = pi.shared(Sr[:100])
            gate_samples = torch.sigmoid(pi.gate_head(h_eval))
        print(f'  Step {step+1:>5d}  Gross SR={sr:.2f}  Net SR={nsr:.2f}  '
              f'σ={s:.3f}  Σ|w|={sw:.2f}  TO={to:.4f}  '
              f'gate=[{gate_samples.mean().item():.3f}±{gate_samples.std().item():.3f}]',
              flush=True)

sr, pn, wm, sw, to, nsr = ev(pi)
torch.save({
    'model_state': pi.state_dict(), 'n_coins': NC, 'latent_dim': DZ,
    'latent_mean': rlm, 'latent_std': rls, 'fee': FEE, 'to': to
}, 'data/rl_policy.pt')
print(f'\nDone: {time.time() - t0:.0f}s', flush=True)
print(f'═══ Final: Gross SR={sr:.2f}  Net SR={nsr:.2f}  Σ|w|={sw:.2f}  TO={to:.4f}', flush=True)
