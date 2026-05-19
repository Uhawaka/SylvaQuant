#!/usr/bin/env python3 -u
"""Complete bias/leakage check for RL experiment returns."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)
ret = d['raw_ret'].astype(np.float32)
spnl = d['sig_pnl'].astype(np.float32)
tr, va, te = d['train_idx'], d['val_idx'], d['test_idx']
ANNUAL = np.sqrt(24*365)

# === 1. Sanity: basic stats ===
print("=== 1. Basic Stats ===")
y_train = ret[tr[0]:tr[1]].mean(1)
y_test = ret[te[0]:te[1]].mean(1)
print(f"Train returns: mean={y_train.mean():+.6f} std={y_train.std():.6f}")
print(f"Test returns:  mean={y_test.mean():+.6f} std={y_test.std():.6f}")
print(f"EW Train SR:   {y_train.mean()/max(y_train.std(),1e-8)*ANNUAL:.4f}")
print(f"EW Test SR:    {y_test.mean()/max(y_test.std(),1e-8)*ANNUAL:.4f}")

# === 2. Check: does exp_data.npz have correct alignment? ===
print("\n=== 2. Verify Data Alignment ===")
# Load VWAP data for BTC to spot-check
vm = np.load(ROOT / 'output/cpcv_vwap_BTCUSDT.npy')
vd = np.load(ROOT / 'output/cpcv_dates_BTCUSDT.npy', allow_pickle=True)
ld = np.load(ROOT / 'data/market_latent_dates.npy', allow_pickle=True)
dm = {pd.Timestamp(d): i for i, d in enumerate(ld)}

# Build full aligned vwap (N)
N_lat = len(ld)
vwap_aligned = np.zeros(N_lat)
for k, dt in enumerate(vd):
    idx = dm.get(pd.Timestamp(dt))
    if idx is not None:
        if idx < N_lat:
            vwap_aligned[idx] = vm[k]

# Check: ret[t] from npz vs manual V[t+2]/V[t+1]-1
print("First 10 bars alignment check (BTC):")
print(f"{'t':>3s}  {'lat[t][0]':>10s}  {'V[t]':>10s}  {'V[t+1]':>10s}  {'V[t+2]':>10s}  {'ret[t](npz)':>12s}  {'V[t+2]/V[t+1]-1':>16s}")
for t in range(10):
    v_cur = vwap_aligned[t] if t < len(vwap_aligned) else 0
    v_nxt = vwap_aligned[t+1] if t+1 < len(vwap_aligned) else 0
    v_nxt2 = vwap_aligned[t+2] if t+2 < len(vwap_aligned) else 0
    r_manual = v_nxt2 / v_nxt - 1 if v_nxt != 0 else 0
    r_npz = ret[t, 0]  # BTC is coin 0
    match = "✓" if abs(r_manual - r_npz) < 1e-8 else "✗"
    print(f"{t:>3d}  {lat[t,0]:>10.4f}  {v_cur:>10.2f}  {v_nxt:>10.2f}  {v_nxt2:>10.2f}  {r_npz:>+12.8f}  {r_manual:>+16.8f}  {match}")

# === 3. RF benchmark ===
print("\n=== 3. RF Predictability (non-linear) ===")
from sklearn.ensemble import RandomForestRegressor
X_train = lat[tr[0]:tr[1]]
X_test = lat[te[0]:te[1]]

# Use 5% of training for speed
n_sub = min(50000, len(X_train))
rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1,
                           random_state=42, verbose=0)
rf.fit(X_train[:n_sub], y_train[:n_sub])
y_pred = rf.predict(X_test)
r2 = 1 - np.mean((y_test - y_pred)**2) / np.var(y_test)
print(f"RF R² (test): {r2:.4f}")
rf_sr = y_pred.mean() / max(y_pred.std(), 1e-8) * ANNUAL
print(f"RF Sharpe (test): {rf_sr:.2f}")

# === 4. Directional accuracy test ===
print("\n=== 4. Directional Accuracy ===")
for n_est in [50, 100]:
    rf2 = RandomForestRegressor(n_estimators=n_est, max_depth=6, n_jobs=-1, random_state=42)
    rf2.fit(X_train[:n_sub], y_train[:n_sub])
    yp = rf2.predict(X_test)
    dir_acc = np.mean((yp > 0) == (y_test > 0))
    print(f"  RF(n_est={n_est}, depth=6): dir_acc={dir_acc:.4f}")

# === 5. Check: is latents ALREADY look-ahead contaminated? ===
print("\n=== 5. Future Information Test ===")
# If latent[t] has no look-ahead, then latent[t] should NOT predict ret[t+1]
# (since ret[t+1] = V[t+3]/V[t+2]-1, and latent[t] only has bar t data)
y_future = ret[1:].mean(1)  # ret[t+1] aligned with latent[t]
X_past = lat[:-1]
rf3 = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
# Train: use first half of training
n_tr = len(X_past) // 2
rf3.fit(X_past[:n_tr], y_future[:n_tr])
print(f"  Predicting ret[t+1] from lat[t]: R²={rf3.score(X_past[n_tr:], y_future[n_tr:]):.4f}")

# === 6. Check: ret_autocorrelation annualized ===
print("\n=== 6. Annualized Momentum Sharpe ===")
for c in range(9):
    r = ret[:, c]
    m = np.isfinite(r)
    rv = r[m]
    if len(rv) > 100:
        ac1 = np.corrcoef(rv[:-1], rv[1:])[0,1]
        # Maximum theoretical SR from autocorrelation trading
        # For a simple strategy: yesterday's return × today's position
        # The Sharpe of this strategy ≈ ac1 * sqrt(N_periods)
        sr_mom = ac1 * np.sqrt(96*365) if ac1 > 0 else 0
        if c == 0:
            print(f"  ret autocorr lag1={ac1:.4f}, mom SR bound={sr_mom:.2f}")
