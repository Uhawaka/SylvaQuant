#!/usr/bin/env python3 -u
"""Check why random NN policy gives SR=3.55 and RF gives SR=7.74."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
ANNUAL = np.sqrt(24*365)

# Normalize (same as run_all.py)
lm = lat[:tr[1]].mean(0, keepdims=True)
ls = lat[:tr[1]].std(0, keepdims=True).clip(1e-6)
lat_n = ((lat - lm) / ls).astype(np.float32)

# Test 1: PyTorch random policy SR vs numpy random
import torch
torch.manual_seed(42)
DZ, NC, H = 16, 9, 128

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(DZ,H), torch.nn.SiLU(),
                                        torch.nn.Linear(H,H), torch.nn.SiLU())
        self.mu = torch.nn.Linear(H,NC)
    def forward(self, s, det=True):
        h = self.net(s)
        m = torch.tanh(self.mu(h))
        return m

pi = Policy()
lat_val = torch.from_numpy(lat_n[va[0]:va[1]])
ret_val = torch.from_numpy(ret[va[0]:va[1]])

# Run 100 different seeds to understand distribution
print("=== PyTorch Random Policy Distribution ===")
seeds = list(range(100))
srs_pt = []
w_means = []
for seed in seeds:
    torch.manual_seed(seed)
    pi2 = Policy()
    with torch.no_grad():
        w = pi2(lat_val)
        pr = (w * ret_val).sum(1).numpy()
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    srs_pt.append(sr)
    w_means.append(w.abs().mean().item())

srs_pt = np.array(srs_pt)
print(f"  SR: mean={srs_pt.mean():.2f} ± {srs_pt.std():.2f}")
print(f"  SR: min={srs_pt.min():.2f} max={srs_pt.max():.2f}")
print(f"  |w| mean: {np.mean(w_means):.4f}")

# Compare with numpy uniform
np.random.seed(42)
srs_np = []
for _ in range(100):
    w = np.random.uniform(-1, 1, (va[1]-va[0], NC))
    pr = (w * ret[va[0]:va[1]]).sum(1)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    srs_np.append(sr)
srs_np = np.array(srs_np)
print(f"\n=== Numpy Uniform Distribution ===")
print(f"  SR: mean={srs_np.mean():.2f} ± {srs_np.std():.2f}")
print(f"  SR: min={srs_np.min():.2f} max={srs_np.max():.2f}")

# Test 2: Check if weight magnitude matters
print(f"\n=== Weight Magnitude Test ===")
for scale in [0.1, 0.5, 1.0, 2.0]:
    np.random.seed(42)
    sr_list = []
    for _ in range(50):
        w = np.random.uniform(-scale, scale, (va[1]-va[0], NC))
        pr = (w * ret[va[0]:va[1]]).sum(1)
        sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
        sr_list.append(sr)
    print(f"  scale={scale:.1f}: SR_mean={np.mean(sr_list):.2f} SR_max={np.max(sr_list):.2f}")

# Test 3: Check RF predictions more carefully
print(f"\n=== RF Prediction Details ===")
from sklearn.ensemble import RandomForestRegressor
X_train = lat_n[tr[0]:tr[1]]
y_train = ret[tr[0]:tr[1]].mean(1)
X_test = lat_n[te[0]:te[1]]
y_test = ret[te[0]:te[1]].mean(1)

rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
rf.fit(X_train[:50000], y_train[:50000])
yp = rf.predict(X_test)

# Distribution of predictions
print(f"  y_pred: mean={yp.mean():+.6f} std={yp.std():.6f}")
print(f"  y_test: mean={y_test.mean():+.6f} std={y_test.std():.6f}")
print(f"  corr(y_pred, y_test)={np.corrcoef(yp, y_test)[0,1]:.4f}")
print(f"  dir_acc={np.mean((yp>0)==(y_test>0)):.4f}")

# The Sharpe calculation:
pr = yp  # portfolio return = prediction * 1 (since it's average across coins)
sr_test = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
print(f"  RF Sharpe (test): {sr_test:.2f}")

# Now what if we use the PREDICTIONS as weights on individual coin returns?
print(f"\n  Using RF as portfolio weights (summed returns, not mean):")
# rf.predict gives mean return prediction. Use as weight on each coin
# weight[t,c] = predict_single_coin_return[t,c] ... but we trained on mean
# Instead, check: what if we use predictions as binary long/short signal?
w_rf = np.zeros((len(y_test), NC))
for c in range(NC):
    y_c = ret[te[0]:te[1], c]
    rf_c = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
    rf_c.fit(X_train[:50000], ret[tr[0]:tr[1], c][:50000])
    yp_c = rf_c.predict(X_test)
    w_rf[:, c] = yp_c

pr_rf = (w_rf * ret[te[0]:te[1]]).sum(1) / NC
sr_rf = pr_rf.mean() / max(pr_rf.std(), 1e-8) * ANNUAL
print(f"  Per-coin RF Sharpe (test): {sr_rf:.2f}")
