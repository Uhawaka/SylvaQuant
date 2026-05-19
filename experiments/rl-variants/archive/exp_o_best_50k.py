#!/usr/bin/env python3 -u
"""
50K steps with best config: Base arch, θ=0.18, seed=1111.
Optimized: fast batch eval during training, full eval only at end.
"""
import sys, warnings, time, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import torch, torch.nn as nn

NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004
SEED = 1111
THETA = 0.18
L = 32
B = 256
BS = 1024  # eval batch size

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat, ret = d['latents'].astype(np.float32), d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
lm, ls = lat[:tr[1]].mean(0, keepdims=True), lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt, rt = torch.from_numpy(lat_n).float(), torch.from_numpy(ret).float()

class DeltaSoftThresh(nn.Module):
    def __init__(self, theta_max=THETA):
        super().__init__()
        self.theta_max = theta_max
        self.encoder = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.SiLU(),
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

class VecEnv:
    def __init__(self, B=B, L=L):
        self.B, self.L = B, L
    def reset(self, split='train'):
        lo, hi = tr if split=='train' else (va if split=='val' else te)
        self.starts = np.random.randint(lo, hi - self.L - 1, size=self.B)
        self.w = torch.zeros(self.B, NC)
        self.t = 0
    def roll_out(self, pi):
        self.reset('train')
        rets = []
        for _ in range(self.L):
            z = lt[self.starts + self.t]
            w = pi(z, self.w)
            r = rt[self.starts + self.t]
            pr = (w * r).sum(1)
            to = (w - self.w).abs().sum(1)
            rets.append(pr - FEE * to)
            self.w = w
            self.t += 1
        return torch.stack(rets, dim=1)

@torch.no_grad()
def eval_fast(pi, split='val'):
    """Batch-sequential eval (fast). Processes BS bars at a time, tracking w_prev."""
    lo, hi = va if split=='val' else te
    z_all = lt[lo:hi]; r_all = rt[lo:hi]
    n = hi - lo
    w = torch.zeros(1, NC)
    all_pr = []
    for i in range(0, n, BS):
        end = min(i + BS, n)
        z_b = z_all[i:end]   # (chunk, DZ)
        r_b = r_all[i:end]   # (chunk, NC)
        ws = []
        for t in range(end - i):
            w = pi(z_b[t:t+1], w)
            ws.append(w)
        w_b = torch.cat(ws, dim=0)
        all_pr.append((w_b * r_b).sum(1))
    pr = torch.cat(all_pr).cpu().numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    return sr

@torch.no_grad()
def eval_full(pi, split='test'):
    """Full sequential eval with TO analysis."""
    lo, hi = te if split=='test' else va
    z = lt[lo:hi]; n = hi - lo
    w = torch.zeros(1, NC)
    pr, ws = [], []
    for i in range(n):
        w = pi(z[i:i+1], w)
        ws.append(w); pr.append((w * rt[lo+i:lo+i+1]).sum().item())
    w_np = torch.cat(ws, dim=0).numpy()
    pr = np.array(pr)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w_np, axis=0)).sum(1).mean()
    fee = np.concatenate([[0.0], FEE * np.abs(np.diff(w_np, axis=0)).sum(1)])
    net = (pr - fee).mean() / max((pr - fee).std(), 1e-8) * ANNUAL
    act = (w_np.sum(1) > 0.001).mean()
    return sr, to, net, act

torch.manual_seed(SEED); np.random.seed(SEED)
pi = DeltaSoftThresh(theta_max=THETA)
opt = torch.optim.AdamW(pi.parameters(), lr=3e-4, weight_decay=1e-5)
env = VecEnv()
t0 = time.time()
STEPS = 50000

print(f'═══ Best config: θ={THETA} L={L} B={B} seed={SEED} {STEPS} steps ═══')
for step in range(STEPS):
    rets = env.roll_out(pi)
    pr_flat = rets.reshape(-1)
    loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
    opt.step()
    if step > 0 and step % 5000 == 0:
        s = eval_fast(pi, 'val')
        print(f'  Step {step:>5d}  Val SR={s:.2f}  wall={time.time()-t0:.0f}s')

print(f'\\nFinal full eval on test...')
s, t, n, a = eval_full(pi, 'test')
sv, _, nv, _ = eval_full(pi, 'val')
print(f'  Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:+.2f}  Act={a:.2%}')
print(f'  Val:  SR={sv:.2f}  Net SR={nv:+.2f}')
print(f'  Total time: {time.time()-t0:.0f}s')
