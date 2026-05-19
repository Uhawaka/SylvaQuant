#!/usr/bin/env python3 -u
"""
Longer training with Cosine LR schedule.
Hypothesis: more steps + better LR schedule → lower variance, higher mean SR.
Tests: 20K, 50K steps with cosine LR.
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
DEV = 'cpu'
THETA = 0.15
L = 32
B = 256

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
        self.split = split
    def _obs(self):
        return lt[self.starts + self.t], self.w
    def step(self, w_new):
        r = rt[self.starts + self.t]
        pr = (w_new * r).sum(1)
        to = (w_new - self.w).abs().sum(1)
        self.w = w_new; self.t += 1
        return pr - FEE * to
    def roll_out(self, pi):
        self.reset('train')
        rets = []
        for _ in range(self.L):
            z, wp = self._obs()
            wn = pi(z, wp)
            rets.append(self.step(wn))
        return torch.stack(rets, dim=1)

@torch.no_grad()
def eval_full(pi, split='test'):
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

def train(pi, STEPS, lr=3e-4, label=''):
    env = VecEnv()
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    val_srs = []
    for step in range(STEPS):
        rets = env.roll_out(pi)
        pr_flat = rets.reshape(-1)
        loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step(); sched.step()
        if step % 5000 == 0 or step == 0:
            s, _, _, _ = eval_full(pi, 'val')
            val_srs.append(s)
            lr_now = sched.get_last_lr()[0]
            print(f'  Step {step:>5d}  Val SR={s:.2f}  lr={lr_now:.2e}  wall={time.time()-t0:.0f}s')
    s, t, n, a = eval_full(pi, 'test')
    print(f'  [{label}] Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:.2f}  Act={a:.2%}')
    return s, t, n, a, val_srs

# ═══ RUN ═══
SEEDS = [42, 123, 456, 789, 1111]
print(f'═══ Longer training sweep ═══')
print(f'Policy: DeltaSoft θ={THETA} L={L} B={B}')
print()

for steps, label in [(20000, '20K'), (50000, '50K')]:
    print(f'{"="*60}')
    print(f'{label} steps — {len(SEEDS)} seeds')
    print(f'{"="*60}')
    outs = []
    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        pi = DeltaSoftThresh()
        s, to, n, a, val_srs = train(pi, STEPS=steps, label=f'{label}-s{seed}')
        outs.append({'seed': seed, 'sr': s, 'to': to, 'net': n, 'act': a})
        print(f'  seed={seed:>4d}  SR={s:>7.2f}  TO={to:.4f}  Net SR={n:>+7.2f}  Act={a:.2%}')
    sr_v = [o['sr'] for o in outs]
    net_v = [o['net'] for o in outs]
    to_v = [o['to'] for o in outs]
    print(f'  ─────────────────────────────────────────────────────')
    print(f'  [{label}] MEAN  SR={np.mean(sr_v):>7.2f}±{np.std(sr_v):.2f}  '
          f'Net SR={np.mean(net_v):>+7.2f}±{np.std(net_v):.2f}  '
          f'TO={np.mean(to_v):.4f}')
    print(f'  [{label}] wall={time.time()-t0:.0f}s')

print(f'\n{"="*60}')
print('DONE')
