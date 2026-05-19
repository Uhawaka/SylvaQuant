#!/usr/bin/env python3 -u
"""
DeltaSoft — delta policy with w[t-1] in state and soft threshold.

w[t] = clamp(w[t-1] + sign(score) · max(|score| - θ, 0), -1, 1)
  state = concat(latent[t], w[t-1])
  score = score_head(encoder(state))
  θ = sigmoid(thresh_head(state)) × θ_max

Key: continuous steps (VecEnv L=32), w[t-1] known → delta=0 → fee=0

Ref: experiments/rl-variants/EXPERIMENT_LOG.md, quant-ml-methodology EventDriven RL
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
DEV = 'cpu'  # CPU: tiny model, MPS kernel overhead kills sequential
DZ, NC, H = 16, 9, 16   # H=16 small, 841 params
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004
T_MAX = 0.15  # theta_max — soft threshold ceiling
LR = 3e-4
B, L = 256, 32  # VecEnv parallel × episode length
N_STEPS = 10000

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
print(f'Data: {lat.shape}  Train: {tr[0]}–{tr[1]}  Val: {va[0]}–{va[1]}  Test: {te[0]}–{te[1]}', flush=True)


class DeltaSoftPolicy(nn.Module):
    """Delta policy: w[t] = clamp(w[t-1] + soft_threshold(score), -1, 1)"""
    def __init__(self):
        super().__init__()
        state_dim = DZ + NC  # latent + w_prev
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)
        self.thresh_head = nn.Linear(H, NC)

    def forward(self, lat_t, w_prev):
        """lat_t: (B, DZ), w_prev: (B, NC) → w: (B, NC)"""
        obs = torch.cat([lat_t, w_prev], dim=-1)  # (B, DZ+NC)
        h = self.encoder(obs)
        score = self.score_head(h)               # unbounded signal
        theta = torch.sigmoid(self.thresh_head(h)) * T_MAX  # learned per-coin threshold
        # Soft threshold on delta
        sign = torch.sign(score)
        mag = (score.abs() - theta).clamp(min=0)
        delta = sign * mag
        # New position
        w = (w_prev + delta).clamp(-1, 1)
        return w

    @torch.no_grad()
    def rollout(self, data_lat, data_ret, lo, hi):
        """Deterministic eval over range. Returns gross_ret, net_ret, TO."""
        n = hi - lo
        w = torch.zeros(1, NC)
        grets, nrets, w_list = [], [], []
        for i in range(n):
            w_new = self.forward(data_lat[lo+i:lo+i+1], w)
            gr = (w_new * data_ret[lo+i:lo+i+1]).sum().item()
            nr = gr - FEE * (w_new - w).abs().sum().item()
            grets.append(gr); nrets.append(nr)
            w_list.append(w_new.clone())
            w = w_new
        ws = torch.cat(w_list, dim=0).numpy()
        to = np.abs(np.diff(ws, axis=0)).sum(1).mean() if n > 1 else 0.0
        return np.array(grets), np.array(nrets), to

    def params(self):
        return sum(p.numel() for p in self.parameters())


# ── VecEnv (CPU) — parallel continuous episodes ──
class VecEnv:
    """B parallel environments, each running L continuous steps."""
    def __init__(self, lat_t, rt_t, train_lo, train_hi):
        self.lat = lat_t
        self.ret = rt_t
        self.lo, self.hi = train_lo, train_hi
        self.T = self.hi - self.lo
        self.w = torch.zeros(B, NC)

    def reset(self):
        """Sample B random start positions. Returns latents, returns for episode."""
        self.w.zero_()
        starts = torch.randint(0, self.T - L - 1, (B,))
        self.idx = starts + self.lo  # (B,) start indices
        return self._get_data()

    def _get_data(self):
        """Get the episode data blocks (B, L, DZ) and (B, L, NC)."""
        idx = self.idx.unsqueeze(1) + torch.arange(L).unsqueeze(0)
        return self.lat[idx], self.ret[idx]

    def roll(self, policy):
        """Run L sequential steps, return (B,) cumulative net portfolio returns."""
        z_seq, r_seq = self._get_data()  # (B, L, DZ), (B, L, NC)
        self.w = self.w.clone()  # detach from previous episode
        cumrets = torch.zeros(B)
        w_prev_det = self.w.clone()

        for t in range(L):
            z_t = z_seq[:, t]  # (B, DZ)
            r_t = r_seq[:, t]  # (B, NC)

            w_t = policy(z_t, w_prev_det)

            # Portfolio return (gross)
            pr = (w_t * r_t).sum(1)

            # Fee: FEE × |Δw|
            to = (w_t - w_prev_det).abs().sum(1)
            net = pr - FEE * to

            cumrets += net

            # Detach w_prev for next step (prevent BPTT through L steps)
            w_prev_det = w_t.detach()

        return cumrets  # (B,) — cumulative net return per env


# ═══════════════════ Trainer ═══════════════════
policy = DeltaSoftPolicy()
print(f'═══ DeltaSoft ═══\nParams: {policy.params()}  B={B} L={L} Steps={N_STEPS} θ_max={T_MAX}')
print(f'State: concat(latent[{DZ}], w_prev[{NC}]) = {DZ+NC}-dim', flush=True)

opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-5)

@torch.no_grad()
def eval_sr(lo, hi):
    grets, nrets, to_ev = policy.rollout(lt, rt, lo, hi)
    gsr = grets.mean() / max(grets.std(), 1e-8) * ANNUAL
    nsr = nrets.mean() / max(nrets.std(), 1e-8) * ANNUAL
    return gsr, nsr, to_ev

# Initial eval
gsr0, nsr0, _ = eval_sr(va[0], va[1])
print(f'Init  Gross SR={gsr0:.2f}  Net SR={nsr0:.2f}', flush=True)

venv = VecEnv(lt, rt, tr[0], tr[1])
t0 = time.time()

for step in range(N_STEPS):
    # Sample episode: reset envs → roll L steps → get cumulative returns
    venv.reset()
    cumrets = venv.roll(policy)  # (B,) cumulative net portfolio return per env

    # Loss = -Sharpe(cumrets)
    sr = cumrets.mean() / max(cumrets.std(), 1e-8)
    loss = -sr

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    opt.step()

    if (step + 1) % 2000 == 0 or step == 0:
        gsr, nsr, to_ev = eval_sr(va[0], va[1])
        print(f'  Step {step+1:>5d}  Gross SR={gsr:.2f}  Net SR={nsr:.2f}  '
              f'wall={time.time()-t0:.0f}s', flush=True)

# Final eval
gsr, nsr, to_ev = eval_sr(te[0], te[1])
print(f'\n═══ FINAL (Test) ═══', flush=True)
print(f'Gross SR={gsr:.2f}  Net SR={nsr:.2f}', flush=True)
print(f'Time: {time.time()-t0:.0f}s', flush=True)

torch.save({'model_state': policy.state_dict()}, str(ROOT / 'data/rl_policy_deltasoft.pt'))
print('Saved data/rl_policy_deltasoft.pt', flush=True)
