#!/usr/bin/env python3 -u
"""Deep diagnostic: why is policy SR still 70+ after alignment fix?"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)  # (189669, 16)
ret = d['raw_ret'].astype(np.float32)  # (189669, 9)
tr, va, te = np.array([0, 132768]), np.array([132768, 161218]), np.array([161218, 189669])
ANNUAL = np.sqrt(24*365)

# Test 1: equal-weight portfolio on test set
ew_test = ret[te[0]:te[1]].mean(1)
sr_ew = ew_test.mean() / max(ew_test.std(), 1e-8) * ANNUAL
print(f"EW Test Sharpe: {sr_ew:.4f}")

# Test 2: random baseline (1000 random policies)
np.random.seed(42)
sr_random = []
for _ in range(1000):
    w = np.random.uniform(-1, 1, (te[1]-te[0], 9))
    pr = (w * ret[te[0]:te[1]]).sum(1)
    sr = pr.mean() / max(pr.std(), 1e-8) * ANNUAL
    sr_random.append(sr)
sr_random = np.array(sr_random)
print(f"Random uniform weights (1000 trials): mean SR={sr_random.mean():.2f} ± {sr_random.std():.2f}")
print(f"  Best: {sr_random.max():.2f}  Worst: {sr_random.min():.2f}")

# Test 3: Check return distribution for skew/kurtosis
print(f"\nReturn distribution per coin:")
for c in range(9):
    r = ret[tr[0]:tr[1], c]
    r = r[np.isfinite(r)]
    print(f"  coin[{c}]: mean={r.mean():+.6f} std={r.std():.6f} "
          f"skew={np.mean(((r-r.mean())/r.std())**3):.4f} "
          f"kurt={np.mean(((r-r.mean())/r.std())**4):.4f} "
          f"max={r.max():+.6f} min={r.min():+.6f}")

# Test 4: Are returns autocorrelated? (momentum test)
print(f"\nReturn autocorrelation (test set):")
for c in range(9):
    r = ret[te[0]:te[1], c]
    m = np.isfinite(r)
    r_clean = r[m]
    ac1 = np.corrcoef(r_clean[:-1], r_clean[1:])[0,1]
    ac2 = np.corrcoef(r_clean[:-2], r_clean[2:])[0,1]
    print(f"  coin[{c}]: lag1={ac1:.4f}  lag2={ac2:.4f}")

# Test 5: How much does VWAP[t] tell us about VWAP[t+2]?
print(f"\nVWAP level persistence (test set):")
for c in range(9):
    r = ret[te[0]:te[1], c]
    m = np.isfinite(r)
    r_clean = r[m]
    if len(r_clean) > 100:
        # Check if latent encodes VWAP level that persists
        # Directional accuracy: sign(latent prediction) vs sign(return)
        pass

# Test 6: Check latent[t] correlation with VWAP[t]
print(f"\nlatent vs VWAP correlation:")
import pandas as pd
ld = np.load(ROOT / 'data/market_latent_dates.npy', allow_pickle=True)
dm = {pd.Timestamp(d): i for i, d in enumerate(ld)}

for c, sym in enumerate(['BTCUSDT','ETHUSDT','SOLUSDT']):
    v = np.load(ROOT / f'output/cpcv_vwap_{sym}.npy')
    vd = np.load(ROOT / f'output/cpcv_dates_{sym}.npy', allow_pickle=True)
    vwap_aligned = np.zeros(len(lat))
    for k, dt in enumerate(vd):
        idx = dm.get(pd.Timestamp(dt))
        if idx is not None:
            vwap_aligned[idx] = v[k]
    
    # lat[t] vs VWAP[t]
    for d_dim in range(4):
        c_val = np.corrcoef(lat[5:-5, d_dim], vwap_aligned[5:-5])[0,1]
        if abs(c_val) > 0.05:
            print(f"  {sym} latent[{d_dim}] vs VWAP[t]: corr={c_val:.4f}")
            break

# Test 7: Check the R² with CORRECT alignment (no look-ahead)
print(f"\nR² sanity check (latent[t] → ret[t] = VWAP[t+2]/VWAP[t+1]-1):")
from sklearn.linear_model import LinearRegression
X_train = lat[:-2][:tr[1]]  # (N-2, 16), same alignment as data
y_train = ret[:tr[1]].mean(1)
X_test = lat[:-2][te[0]:te[1]]
y_test = ret[te[0]:te[1]].mean(1)

lr = LinearRegression().fit(X_train[:tr[1]//2], y_train[:tr[1]//2])
print(f"  Linear R² (test): {lr.score(X_test, y_test):.4f}")

# Test 8: What if we use a non-linear model (RF)?
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
rf.fit(X_train[:50000], y_train[:50000])
print(f"  RF R² (test): {rf.score(X_test, y_test):.4f}")
rf_preds = rf.predict(X_test)
rf_sr = rf_preds.mean() / max(rf_preds.std(), 1e-8) * ANNUAL
print(f"  RF Sharpe (test): {rf_sr:.2f}")

# Test 9: Single-coin weight test
print(f"\nSingle-coin equal-weight SR:")
for c in range(9):
    r = ret[te[0]:te[1], c]
    sr_c = r.mean() / max(r.std(), 1e-8) * ANNUAL
    print(f"  coin[{c}]: SR={sr_c:.4f}")
