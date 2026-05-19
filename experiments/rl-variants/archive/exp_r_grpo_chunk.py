#!/usr/bin/env python3 -u
"""
GRPO — Path-level reward (累积 return over L steps, group norm across B envs).

架构:
  B 个 parallel env，每个跑 L 步连续时序
  每步 policy(z) → w，env 内部累积收益 cum_ret += w·ret
  L 步结束后: 每个 env 得到 total_cumret = Σ_{l=1..L} (w_l · ret_l)
  GRPO: 在 B 个 env 的 total_cumret 上 group norm → advantage
  每步 log_prob 都乘同一个 env-level advantage → PG

关键: reward 是 path 内累积值，group 是 B 个并行 env，不是 K 个 action 采样。
"""
import sys, time, warnings, numpy as np, math
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import torch, torch.nn as nn, torch.nn.functional as F

SEED = 1111
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DZ, NC, H = 16, 9, 64
LR = 3e-4; B, L = 2048, 10; N_STEPS = 3000
ANNUAL = np.sqrt(24 * 365); FEE = 0.0004

torch.manual_seed(SEED); np.random.seed(SEED)

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt_t = torch.from_numpy(lat_n).to(DEV)
rt_t = torch.from_numpy(ret).to(DEV)
T = tr[1] - tr[0]  # train length
N_seg = T - L + 1   # number of possible segments

print(f'Data: {lat.shape}  Train: {T} bars', flush=True)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu_head = nn.Linear(H, NC)
        self.log_std = nn.Parameter(torch.zeros(NC))

    def forward(self, z, det=False):
        h = self.shared(z); mu = torch.tanh(self.mu_head(h))
        return mu if det else torch.tanh(mu + torch.randn_like(mu) * self.log_std.exp())

    def act(self, z):
        """Sample one action per state, return (w, log_prob)."""
        h = self.shared(z); mu = torch.tanh(self.mu_head(h))
        std = self.log_std.exp().expand_as(mu)
        eps = torch.randn_like(mu); w = torch.tanh(mu + eps * std)
        # Log prob of tanh-squashed Gaussian
        lp = -0.5*(eps**2 + 2*self.log_std + math.log(2*math.pi))
        lp = lp - (2*(math.log(2)-(mu+eps*std)-F.softplus(-2*(mu+eps*std))))
        return w, lp.sum(-1)  # (B, NC), (B,)


# ═══════════════════ PathEnv ═══════════════════
class PathEnv:
    """B parallel envs, each running L consecutive bars as one path."""
    def __init__(self, start_idx):
        """
        start_idx: (B,) starting indices for each path.
        B contiguous paths of length L, all starting from start_idx.
        """
        offset = torch.arange(L, device=DEV).view(1, L)  # (1, L)
        sidx = start_idx.view(-1, 1)  # (B, 1)
        self.z = lt_t[sidx + offset]   # (B, L, DZ)
        self.r = rt_t[sidx + offset]   # (B, L, NC)
        self.t = 0
        self.w_prev = torch.zeros(B, NC, device=DEV)

    def step(self):
        """Return current state for each env. (B, DZ)"""
        z_t = self.z[:, self.t]
        return z_t

    def close_step(self, w):
        """
        After receiving all actions for this step, advance.
        Returns: reward contribution (w·ret) for this step.
        """
        r_t = self.r[:, self.t]  # (B, NC)
        rets = (w * r_t).sum(1)  # (B,) portfolio return
        self.w_prev = w.clone()
        self.t += 1
        return rets


@torch.no_grad()
def eval_full(pi, lo, hi):
    z = torch.from_numpy(lat_n[lo:hi]).to(DEV)
    r = torch.from_numpy(ret[lo:hi]).to(DEV)
    n = hi - lo
    w = torch.zeros(1, NC, device=DEV)
    grets, nrets, ws = [], [], []
    for i in range(n):
        wn = pi(z[i:i+1], det=True)
        ws.append(wn.cpu().numpy().ravel())
        gr = (wn * r[i:i+1]).sum().item()
        nr = gr - FEE * (wn - w).abs().sum().item()
        grets.append(gr); nrets.append(nr)
        w = wn
    ws = np.array(ws)
    pr, nr = np.array(grets), np.array(nrets)
    gsr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    nsr = nr.mean() / max(nr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(ws, axis=0)).sum(1).mean() if n > 1 else 0.0
    return gsr, nsr, to, np.abs(ws).mean()


# ═══════════════════ TRAIN ═══════════════════
pi = Policy().to(DEV)
opt = torch.optim.AdamW(pi.parameters(), lr=LR)
print(f'B={B} L={L} Steps={N_STEPS} Params={sum(p.numel() for p in pi.parameters())}', flush=True)

gsr0, nsr0, to0, wa0 = eval_full(pi, va[0], va[1])
print(f'Init  Gross SR={gsr0:.2f}  Net SR={nsr0:.2f}  TO={to0:.4f}  |w|={wa0:.3f}', flush=True)

t0 = time.time()
for step in range(N_STEPS):
    # Sample B random starts from training range (prevent overlap = better diversity)
    starts = torch.randint(0, N_seg, (B,), device=DEV) + tr[0]
    env = PathEnv(starts)

    # Storage for per-step log_probs, states, and cumulative returns
    states = torch.zeros(L, B, DZ, device=DEV)
    ws = torch.zeros(L, B, NC, device=DEV)
    lps = torch.zeros(L, B, device=DEV)
    cumrets = torch.zeros(B, device=DEV)

    for l in range(L):
        z = env.step()         # (B, DZ)
        w, lp = pi.act(z)      # (B, NC), (B,)
        ret_contrib = env.close_step(w)
        states[l] = z
        ws[l] = w
        lps[l] = lp
        cumrets += ret_contrib

    # Group norm: normalize across B envs
    adv = (cumrets - cumrets.mean()) / (cumrets.std() + 1e-8)  # (B,)

    # PG: all L steps share same env-level advantage
    pg_loss = -(lps * adv.unsqueeze(0)).mean()
    entropy = -lps.mean()

    # KL: average across steps, compute from stored states
    h_all = pi.shared(states.view(-1, DZ))  # (L*B, H)
    mu_all = torch.tanh(pi.mu_head(h_all)).view(L, B, NC)
    kl = 0.5 * (mu_all**2 + pi.log_std.exp()**2 - 1 - 2 * pi.log_std).mean()

    loss = pg_loss + 0.001 * entropy + 0.05 * kl

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    opt.step()

    if (step + 1) % 500 == 0:
        gsr, nsr, to_ev, wa = eval_full(pi, va[0], va[1])
        s = pi.log_std.exp().mean().item()
        print(f'Step {step+1:>4d}  Gross SR={gsr:.2f}  Net SR={nsr:.2f}  TO={to_ev:.4f}  |w|={wa:.3f}  σ={s:.3f}  cumret_mean={cumrets.mean().item():.6f}  wall={time.time()-t0:.0f}s', flush=True)

gsr, nsr, to_ev, wa = eval_full(pi, te[0], te[1])
print(f'\n═══ TEST ═══')
print(f'Gross SR={gsr:.2f}  Net SR={nsr:.2f}  TO={to_ev:.4f}  |w|={wa:.3f}', flush=True)
print(f'Time: {time.time()-t0:.0f}s', flush=True)
torch.save({'model_state': pi.state_dict()}, str(ROOT / 'data/rl_policy_grpo.pt'))
print('Saved', flush=True)
