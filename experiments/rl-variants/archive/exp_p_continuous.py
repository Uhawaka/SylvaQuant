#!/usr/bin/env python3 -u
"""
Continuous time-slice training for DeltaSoft RL.

Problem: VecEnv resets to random start positions every step.
  → Policy never experiences >L bars of continuous trading
  → No consequence for holding bad positions
  → Turnover cost is only L-step local, not long-term

Fix: Train on large contiguous chunks of the time series.
  Process 4000+ bars sequentially, Sharpe over entire chunk.
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
L_chunk = 4096  # chunk size: ~42 days of 15m data
STRIDE = 2048   # 50% overlap between chunks

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)
lt = torch.from_numpy(lat_n).float()
rt = torch.from_numpy(ret).float()

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

# ═══════════════════════════════════════════
# ChunkEnv: Continuous time-slice environment
# ═══════════════════════════════════════════
class ChunkEnv:
    """
    Continuous chunk environment.
    
    Unlike VecEnv which resets to random starts each step,
    ChunkEnv processes one long continuous sequence at a time.
    
    Flow:
      1. Select chunk of L bars from training range
      2. Run policy sequentially through all L bars
      3. Return (B,) returns for Sharpe computation
      
    w_prev carries over naturally — no artificial resets.
    """
    def __init__(self, lt, rt, tr, chunk_size=4096):
        self.lt = lt
        self.rt = rt
        self.lo, self.hi = tr  # training range
        self.chunk_size = chunk_size
        self.max_start = max(0, self.hi - self.lo - chunk_size)
    
    def sample_chunk(self):
        """Pick a random contiguous chunk from training data."""
        start = self.lo + np.random.randint(0, max(1, self.max_start))
        return start, start + self.chunk_size
    
    def run_episode(self, pi, lo, hi):
        """
        Run policy sequentially through [lo, hi).
        Returns portfolio returns with fees.
        """
        w = torch.zeros(1, NC)
        returns = []
        z_seq = self.lt[lo:hi]
        r_seq = self.rt[lo:hi]
        
        for t in range(hi - lo):
            w_new = pi(z_seq[t:t+1], w)
            r = r_seq[t:t+1]
            pr = (w_new * r).sum()
            to = (w_new - w).abs().sum()
            returns.append(pr - FEE * to)
            w = w_new
        
        return torch.stack(returns)

# ── Eval ──
@torch.no_grad()
def eval_sequential(pi, lo, hi):
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

# ═══════════════════════════════════════════
# Train
# ═══════════════════════════════════════════
def train(pi, STEPS=5000, lr=3e-4, chunk_size=4096, verbose=True):
    env = ChunkEnv(lt, rt, tr, chunk_size=chunk_size)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    
    for step in range(STEPS):
        lo, hi = env.sample_chunk()
        rets = env.run_episode(pi, lo, hi)  # (L,)
        
        loss = -rets.mean() / (rets.std() + 1e-8)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
        opt.step()
        
        if verbose and (step % 1000 == 0 or step == 0):
            s, _, _, _ = eval_sequential(pi, va[0], va[1])
            print(f'  Step {step:>5d}  Val SR={s:.2f}  wall={time.time()-t0:.0f}s')
    
    if verbose:
        print('  Final eval...')
    s, t, n, a = eval_sequential(pi, te[0], te[1])
    sv, _, nv, _ = eval_sequential(pi, va[0], va[1])
    print(f'  Test: SR={s:.2f}  TO={t:.4f}  Net SR={n:+.2f}  Act={a:.2%}')
    print(f'  Val:  SR={sv:.2f}  Net SR={nv:+.2f}')
    return {'test_sr': s, 'test_to': t, 'test_net': n, 'test_act': a}

# ═══════════════════ RUN ═══════════════════
torch.manual_seed(SEED); np.random.seed(SEED)
print('═══ Continuous Chunk Training ═══')
print(f'Chunk size: {L_chunk} bars ({L_chunk*15/60/24:.1f} days of 15m data)')
print(f'Train range: {tr}, Val range: {va}, Test range: {te}')
print()

for chunk_size in [1024, 2048, 4096]:
    pi = DeltaSoftThresh(theta_max=THETA)
    r = train(pi, STEPS=3000, chunk_size=chunk_size)
    print(f'  chunk={chunk_size}: SR={r["test_sr"]:.2f}  Net SR={r["test_net"]:+.2f}  TO={r["test_to"]:.4f}')
