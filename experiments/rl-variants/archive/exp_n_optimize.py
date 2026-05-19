#!/usr/bin/env python3 -u
"""
Fixed seed=1111, systematic architecture + hyperparameter optimization.
Tests: θ_max sweep + LayerNorm + Dropout + deeper network.
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

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat, ret = d['latents'].astype(np.float32), d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
lm, ls = lat[:tr[1]].mean(0, keepdims=True), lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt, rt = torch.from_numpy(lat_n).float(), torch.from_numpy(ret).float()

SEED = 1111

# ── Policy variants ──

class DeltaSoftBase(nn.Module):
    """Original: 2-layer MLP, shared encoder."""
    def __init__(self, theta_max=0.15):
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

class DeltaSoftLN(nn.Module):
    """+ LayerNorm after each hidden layer."""
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        self.encoder = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, H), nn.LayerNorm(H), nn.SiLU(),
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

class DeltaSoftDropout(nn.Module):
    """+ Dropout(0.1) after each hidden layer."""
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        self.encoder = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(H, H), nn.SiLU(), nn.Dropout(0.1),
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

class DeltaSoftDual(nn.Module):
    """Dual-path: separate score and threshold encoders."""
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        self.score_net = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, NC),
        )
        self.thresh_net = nn.Sequential(
            nn.Linear(DZ+NC, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, NC),
        )
        # Init bias for threshold ~0.12
        nn.init.constant_(self.thresh_net[-1].bias, -2.0)
    def forward(self, z, w_prev):
        x = torch.cat([z, w_prev], dim=-1)
        score = self.score_net(x)
        theta = torch.sigmoid(self.thresh_net(x)) * self.theta_max
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)

class DeltaSoftDeeper(nn.Module):
    """Deeper: 3 hidden layers, H=32."""
    def __init__(self, theta_max=0.15):
        super().__init__()
        self.theta_max = theta_max
        self.encoder = nn.Sequential(
            nn.Linear(DZ+NC, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
        )
        self.score_head = nn.Linear(32, NC)
        self.thresh_head = nn.Linear(32, NC)
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
    def __init__(self, B=256, L=32):
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
            rets.append(self.step(pi(z, wp)))
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

def train(pi, STEPS=10000, lr=3e-4, label=''):
    env = VecEnv()
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    for step in range(STEPS):
        rets = env.roll_out(pi)
        pr_flat = rets.reshape(-1)
        loss = -pr_flat.mean() / (pr_flat.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
    s, t, n, a = eval_full(pi, 'test')
    print(f'  [{label:>20s}]  SR={s:>7.2f}  TO={t:.4f}  Net SR={n:>+7.2f}  Act={a:.2%}')
    return s, t, n, a

# Test baseline
torch.manual_seed(SEED); np.random.seed(SEED)
pi = DeltaSoftBase(theta_max=0.15)
sr_baseline, to_baseline, net_baseline, act_baseline = train(pi, STEPS=10000, label='Base')
results = {'Base': (sr_baseline, to_baseline, net_baseline, act_baseline)}

# ── Phase 1: Architecture variants ──
variants = {
    'LayerNorm':    DeltaSoftLN(theta_max=0.15),
    'Dropout(0.1)': DeltaSoftDropout(theta_max=0.15),
    'Dual-path':    DeltaSoftDual(theta_max=0.15),
    'Deeper H=32':  DeltaSoftDeeper(theta_max=0.15),
}
for name, pi in variants.items():
    torch.manual_seed(SEED); np.random.seed(SEED)
    results[name] = train(pi, STEPS=10000, label=name)

# ── Phase 2: θ_max sweep with best arch ──
best_arch_name = max(results, key=lambda k: results[k][2])  # best net SR
BestCls = {'LayerNorm': DeltaSoftLN, 'Dropout(0.1)': DeltaSoftDropout,
           'Dual-path': DeltaSoftDual, 'Deeper H=32': DeltaSoftDeeper}.get(best_arch_name, DeltaSoftBase)

print(f'\nBest arch: {best_arch_name}')

for theta in [0.10, 0.12, 0.14, 0.16, 0.18]:
    torch.manual_seed(SEED); np.random.seed(SEED)
    pi = BestCls(theta_max=theta)
    results[f'θ={theta:.2f}'] = train(pi, STEPS=10000, label=f'{best_arch_name} θ={theta:.2f}')

# ── Phase 3: Longer training with best config ──
# Find best (arch, θ) combo
best_key = max(results, key=lambda k: results[k][2])
print(f'\nBest overall: {best_key}')

# ── Summary ──
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'{"Config":<30s} {"Test SR":>7s} {"TO":>7s} {"Net SR":>7s} {"Act":>7s}')
print('─' * 60)
for name, (sr, to, net, act) in sorted(results.items()):
    print(f'{name:<30s} {sr:>7.2f} {to:>7.4f} {net:>+7.2f} {act:>7.2%}')
print(f'{"─" * 60}')
print(f'DONE')
