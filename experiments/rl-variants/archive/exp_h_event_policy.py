#!/usr/bin/env python3 -u
"""
Event-Driven Trading Policy — "trade when" RL.
  w[t] = (1 - α·e[t])·w[t-1] + α·e[t]·d[t]

When e[t]≈0: hold position → zero fee
When e[t]≈1: adjust toward new direction d[t]

Captures the factor-style event→hold→end structure in differentiable RL.
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

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NC, DZ, H = 9, 16, 16
ANNUAL = np.sqrt(24 * 365)
FEE = 0.0004

# ── Data ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']

lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

lt = torch.from_numpy(lat_n).to(DEV)
rt = torch.from_numpy(ret).to(DEV)

# ── Event Policy ──
class EventPolicy(nn.Module):
    """
    Event-driven trading with stateful positions.
    
    w[t] = (1 - α·e[t])·w[t-1] + α·e[t]·d[t]
    
    e[t] = event_strength ∈ [0, 1]  — sigmoid output
    d[t] = direction ∈ [-1, 1]       — tanh output
    α = update_rate ∈ (0, 1)         — learnable global parameter
    
    When e[t]≈0: hold position (zero TO cost)
    When e[t]≈1: rotate (1-α) of old + α of new direction
    """
    def __init__(self, alpha_init=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(DZ, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU()
        )
        self.dir_head = nn.Linear(H, NC)    # direction d[t]
        self.event_head = nn.Linear(H, NC)  # event strength e[t]
        
        # Learnable α (sigmoid-constrained to (0,1))
        self.logit_alpha = nn.Parameter(torch.tensor(
            np.log(alpha_init / (1 - alpha_init)), dtype=torch.float32
        ))
        
        # Init: event starts low (most bars = no trade)
        nn.init.constant_(self.event_head.bias, -1.5)  # sigmoid(-1.5) ≈ 0.18
        
    def forward(self, s, w_prev=None):
        """Forward pass. w_prev: (B, NC) or None.
        
        During training with random batches, w_prev can be approximated
        by the previous bar's weights in the batch: w[batch_i-1].
        """
        h = self.encoder(s)
        d = torch.tanh(self.dir_head(h))      # direction ∈ [-1, 1]
        e = torch.sigmoid(self.event_head(h))  # event ∈ [0, 1]
        a = torch.sigmoid(self.logit_alpha)    # α ∈ (0, 1)
        
        if w_prev is None:
            w_prev = torch.zeros_like(d)
        
        # w[t] = (1-α·e)·w[t-1] + α·e·d[t]
        alpha_e = a * e
        w = (1 - alpha_e) * w_prev + alpha_e * d
        
        return w, {'event': e, 'direction': d, 'alpha': a, 'alpha_e': alpha_e}

# ── Baseline: Soft Threshold (for comparison) ──
class SoftThreshPolicy(nn.Module):
    def __init__(self, theta_max=0.3):
        super().__init__()
        self.theta_max = theta_max
        self.net = nn.Sequential(nn.Linear(DZ,H), nn.SiLU(), nn.Linear(H,H), nn.SiLU())
        self.score = nn.Linear(H, NC)
        self.thresh = nn.Linear(H, NC)
        nn.init.constant_(self.thresh.bias, -2.5)
    def forward(self, s):
        h = self.net(s)
        score = self.score(h)
        theta = torch.sigmoid(self.thresh(h)) * self.theta_max
        abs_s = score.abs()
        w = score.sign() * (abs_s - theta).clamp(min=0)
        return w, {'theta': theta, 'active': (abs_s > theta).float().mean()}

# ── Evaluation ──
@torch.no_grad()
def evaluate(pi, ret_src, split='test', stateful=True):
    """
    Evaluate policy on val/test split.
    For EventPolicy: runs sequential (stateful), tracking w_prev.
    For SoftThresh: independent per bar (no state needed).
    """
    lo, hi = va if split == 'val' else te
    n = hi - lo
    
    w_all = []
    w_prev = torch.zeros(1, NC, device=DEV)
    event_means = []
    
    for i in range(n):
        s_t = lt[lo + i: lo + i + 1]
        
        if stateful:
            w_t, info = pi(s_t, w_prev)
            w_prev = w_t
        else:
            w_t, info = pi(s_t)
        
        w_all.append(w_t)
        if 'event' in info:
            event_means.append(info['event'].mean().item())
    
    w = torch.cat(w_all, dim=0)  # (n, 9)
    ret_slice = ret_src[lo:hi] if isinstance(ret_src, torch.Tensor) else rt[lo:hi]
    pr = (w * ret_slice).sum(1).cpu().numpy()
    
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    to = np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1).mean()
    
    # Fee-adjusted
    fee_cost = FEE * np.abs(np.diff(w.cpu().numpy(), axis=0)).sum(1)
    fee_cost = np.concatenate([[0.0], fee_cost])
    pr_fee = pr - fee_cost
    sr_fee = pr_fee.mean() / max(pr_fee.std(), 1e-8) * ANNUAL
    
    active = (w.abs().sum(1).cpu().numpy() > 0.001).mean()
    avg_w = w.abs().mean().item()
    avg_event = np.mean(event_means) if event_means else 0
    
    return sr, to, sr_fee, active, avg_w, avg_event, pr, w

# ── Training ──
def train(pi, ret_src, name, STEPS=20000, B=4096, lr=3e-4, stateful=True):
    print(f'\n{"="*60}')
    print(f'[Train] {name}')
    print(f'{"="*60}')
    
    pi.to(DEV)
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=1e-5)
    best = {'sr': -10}
    t0 = time.time()
    
    for step in range(STEPS):
        # Sample contiguous segment for stateful training
        ts = np.random.randint(tr[0], tr[1] - B - 1)
        z = lt[ts:ts + B]
        r = rt[ts:ts + B] if ret_src == 'raw' else st[ts:ts + B]
        
        if stateful:
            # Sequential position update within segment
            w = []
            w_prev = torch.zeros(1, NC, device=DEV)
            event_vals = []
            for t in range(B):
                w_t, info = pi(z[t:t+1], w_prev)
                w.append(w_t)
                w_prev = w_t
                if 'event' in info:
                    event_vals.append(info['event'].mean().item())
            w = torch.cat(w, dim=0)
            avg_e = np.mean(event_vals) if event_vals else 0
            avg_a_e = torch.sigmoid(pi.logit_alpha).item() if hasattr(pi, 'logit_alpha') else 0
        else:
            w, info = pi(z)
            avg_e = info.get('active', torch.tensor(0)).item() if isinstance(info, dict) else 0
            avg_a_e = 0
        
        # Fee-aware loss
        w_prev_batch = torch.cat([w[:1], w[:-1]], dim=0)
        to_pen = FEE * (w - w_prev_batch).abs().sum(1)
        pr = (w * r).sum(1) - to_pen
        
        loss = -pr.mean() / (pr.std() + 1e-8)
        
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 0.5); opt.step()
        
        if step % 5000 == 0 or step == 0:
            sr_v, to_v, _, act_v, aw_v, ev_v, _, _ = evaluate(pi, rt, 'val', stateful=stateful)
            pi.train()
            if sr_v > best['sr']:
                best['sr'] = sr_v
            print(f'  Step {step:>5d}  Val SR={sr_v:.2f}  TO={to_v:.4f}  '
                  f'Act={act_v:.2%}  |w|={aw_v:.3f}  e≈{ev_v:.4f}')
    
    # Final test eval
    sr_t, to_t, net_t, act_t, aw_t, ev_t, pr_t, w_t = evaluate(pi, rt, 'test', stateful=stateful)
    a_str = f"  α={avg_a_e:.3f}" if 'avg_a_e' in dir() else ""
    print(f'  [{name}] Done in {time.time()-t0:.0f}s')
    print(f'  Test: SR={sr_t:.2f}  TO={to_t:.4f}  Net SR={net_t:.2f}  '
          f'Act={act_t:.2%}  |w|={aw_t:.3f}  e≈{ev_t:.4f}{a_str}')
    
    return {'test': sr_t, 'to': to_t, 'net': net_t, 'active': act_t, 'avg_w': aw_t, 'event': ev_t, 'pr': pr_t, 'w': w_t}

# ════════════════ RUN ════════════════
results = {}
t_all = time.time()

# 1. Event Policy (α_init=0.3 — moderate update rate)
print(f'\n═══ Event-Driven Policy — "Trade When" RL ═══')
results['event_a03'] = train(EventPolicy(alpha_init=0.3), 'raw', 'Event α_init=0.3', stateful=True)

# 2. Event Policy (α_init=0.1 — slower updates, longer holds)
results['event_a01'] = train(EventPolicy(alpha_init=0.1), 'raw', 'Event α_init=0.1', stateful=True)

# 3. Event Policy (α_init=0.5 — faster updates, shorter holds)
results['event_a05'] = train(EventPolicy(alpha_init=0.5), 'raw', 'Event α_init=0.5', stateful=True)

# 4. Soft Threshold (baseline, already proven)
print(f'\n═══ Baseline: Soft Threshold ═══')
# Quick retrain for fair comparison
st_pi = SoftThreshPolicy(theta_max=0.3).to(DEV)
opt = torch.optim.AdamW(st_pi.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(10000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w, info = st_pi(z)
    w_prev = torch.cat([w[:1], w[:-1]]); to_p = FEE * (w-w_prev).abs().sum(1)
    pr = (w*r).sum(1) - to_p
    loss = -pr.mean()/(pr.std()+1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(st_pi.parameters(), 0.5); opt.step()

sr, to, net, act, aw, _, _, _ = evaluate(st_pi, rt, 'test', stateful=False)
results['softthresh'] = {'test': sr, 'to': to, 'net': net, 'active': act, 'avg_w': aw, 'event': 0}

# ── Analysis: Event Policy Behavior ──
print(f'\n{"="*60}')
print('[Analysis] Event Policy Behavior (test set)')
print(f'{"="*60}')

# Load best event policy for analysis
pi_e = EventPolicy(alpha_init=0.3).to(DEV)
opt = torch.optim.AdamW(pi_e.parameters(), lr=3e-4, weight_decay=1e-5)
for _ in range(15000):
    ts = np.random.randint(tr[0], tr[1] - 4096 - 1)
    z = lt[ts:ts+4096]; r = rt[ts:ts+4096]
    w = []; wp = torch.zeros(1, NC, device=DEV)
    for t in range(4096):
        wt, _ = pi_e(z[t:t+1], wp); w.append(wt); wp = wt
    w = torch.cat(w, dim=0)
    wp2 = torch.cat([w[:1], w[:-1]])
    tp = FEE * (w-wp2).abs().sum(1)
    pr2 = (w*r).sum(1) - tp
    loss = -pr2.mean()/(pr2.std()+1e-8)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(pi_e.parameters(), 0.5); opt.step()

# Full test evaluation with recording
lo, hi = te
w_all = []; e_all = []; d_all = []
wp = torch.zeros(1, NC, device=DEV)
for i in range(hi - lo):
    wt, info = pi_e(lt[lo+i:lo+i+1], wp)
    w_all.append(wt)
    e_all.append(info['event'].cpu().numpy())
    d_all.append(info['direction'].cpu().numpy())
    wp = wt

w = torch.cat(w_all, dim=0)
events = np.concatenate(e_all, axis=0)
directions = np.concatenate(d_all, axis=0)
alpha = torch.sigmoid(pi_e.logit_alpha).item()

print(f'  Learned α: {alpha:.4f}')
print(f'  Event strength: mean={events.mean():.4f}  std={events.std():.4f}')
print(f'  Event > 0.5: {np.mean(events > 0.5):.2%} of bars')
print(f'  Event = 0 (exact): {np.mean(events < 0.001):.2%} of bars')

# Distribution of event per coin
print(f'\n  Per-coin event stats:')
for c in range(NC):
    e_c = events[:, c]
    print(f'    coin[{c}]: mean={e_c.mean():.4f}  >0.5={np.mean(e_c > 0.5):.2%}')

# Position change analysis: how many bars between position changes?
pos_changes = np.sum(np.abs(np.diff(w.cpu().numpy(), axis=0)) > 0.001, axis=1)
print(f'\n  Position changes per bar: mean={pos_changes.mean():.3f} coins')
print(f'  Avg length of "hold" periods (bars between events):')

# Consecutive zero-event bars per coin
for c in range(NC):
    e_binary = (events[:, c] > 0.5).astype(int)
    transitions = np.diff(np.concatenate([[0], e_binary, [0]]))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    if len(starts) > 0 and len(ends) > 0:
        hold_lengths = ends - starts
        if len(hold_lengths) > 0:
            print(f'    coin[{c}]: {len(hold_lengths)} events, avg hold={hold_lengths.mean():.1f} bars')

# ── Summary ──
print(f'\n{"="*60}')
print('FINAL SUMMARY — Event-Driven vs Previous Best')
print(f'{"="*60}')
print(f'{"Method":<25s} {"Test SR":>7s} {"TO":>7s} {"Net SR":>7s} {"Active":>7s} {"e(event)":>8s}')
print('-' * 64)
for name, r in sorted(results.items()):
    e_str = f'{r["event"]:.4f}' if 'event' in r and r['event'] > 0 else '—'
    print(f'{name:<25s} {r["test"]:>7.2f} {r["to"]:>7.4f} {r["net"]:>7.2f} {r["active"]:>7.2%} {e_str:>8s}')

print(f'\nTotal: {time.time()-t_all:.0f}s')
