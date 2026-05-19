#!/usr/bin/env python3 -u
"""Deep dive into leakage source."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent.parent

# ── 1. Check: is the VWAP alignment correct? ──
print('── 1. Timestamp alignment check ──')
dates=np.load(str(ROOT/'data/market_latent_dates.npy'),allow_pickle=True)
print(f'market_latent_dates: {len(dates)} from {str(dates[0])} to {str(dates[-1])}')

# Check BTC VWAP dates alignment
btc_vwap=np.load(str(ROOT/'output/cpcv_vwap_BTCUSDT.npy'))
btc_dates=np.load(str(ROOT/'output/cpcv_dates_BTCUSDT.npy'))
print(f'BTC VWAP: {len(btc_vwap)} from {str(btc_dates[0])} to {str(btc_dates[-1])}')

# Check alignment: does market_latent_dates[i] == cpcv_dates[i] for early bars?
dm={pd.Timestamp(d):i for i,d in enumerate(dates)}
import pandas as pd
btc_idx=[dm.get(pd.Timestamp(d)) for d in btc_dates[:10]]
print(f'BTC first 10 dates aligned indices: {btc_idx}')
print(f'  (If all 0-9, alignment bar-to-bar is correct)')

# ── 2. Direct VWAP leakage test ──
print('\n── 2. VWAP[t] in features? ──')
# The market_latent AE was trained on features that include log_vol[t], log_qv[t]
# VWAP[t] = quote_vol[t]/volume[t] = exp(log_qv[t]-log_vol[t])
# Does this mean latent[t] contains VWAP[t]?
# If so, check if VWAP[t+2]/VWAP[t+1]-1 is predictable from VWAP[t] alone
vm=np.load(str(ROOT/'output/cpcv_vwap_BTCUSDT.npy'))
r2=vm[2:]/vm[1:-1]-1
# Lag-2 autocorrelation: does VWAP ret[t] correlate with VWAP[t] level?
ac=np.corrcoef(np.log(vm[2:-2]),np.sign(r2[1:]))[0,1] if len(vm)>4 else 0
print(f'  VWAP[t] level vs VWAP[t+2]/VWAP[t+1]-1 direction: corr={ac:.4f}')

# ── 3. Check: does close[t] predict VWAP[t+2]/VWAP[t+1]-1? ──
print('\n── 3. Close vs VWAP return timing ──')
# If features use close[t] and VWAP[t+2] is correlated with close[t]...
# Check autocorrelation of returns
for j,sym in enumerate(['BTCUSDT','ETHUSDT','SOLUSDT']):
    v=np.load(str(ROOT/f'output/cpcv_vwap_{sym}.npy'))
    r=v[2:]/v[1:-1]-1
    # autocorrelation at lag 0 (contemporaneous) between close[t] and ret[t]
    # Actually, just check: does VWAP[t+2] correlate with VWAP[t]?
    c=np.corrcoef(v[:-4],v[2:-2])[0,1]
    print(f'  {sym}: corr(VWAP[t], VWAP[t+2])={c:.4f}')
    # This tells us how much price level persists over 2 bars

# ── 4. Direct test: is the latent encoding something trivial? ──
print('\n── 4. What does the latent encode? ──')
lat=np.load(str(ROOT/'data/market_latent.npy'))
# Check first few dimensions
for d in range(4):
    l=lat[:,d]
    print(f'  latent[{d}]: mean={l.mean():.4f} std={l.std():.4f} min={l.min():.4f} max={l.max():.4f}')
    # Autocorrelation
    if len(l)>100:
        ac1=np.corrcoef(l[:-1],l[1:])[0,1]
        ac5=np.corrcoef(l[:-5],l[5:])[0,1]
        print(f'           autocorr lag1={ac1:.4f} lag5={ac5:.4f}')

# ── 5. The smoking gun: does the linear prediction work because of leak? ──
print('\n── 5. Time-shift test ──')
# If no leakage: permuting returns randomly should give R²≈0
# If leakage: permuted returns still give R²≈0 (leak is in timing, not data)
from sklearn.linear_model import LinearRegression
tr=[0,132768]
X=lat[:-2][tr[0]:tr[1]];y_raw=np.load(str(ROOT/'data/rl_exp/exp_data.npz'))['raw_ret'][tr[0]:tr[1]].mean(1)
# Shuffle returns
np.random.seed(42)
y_shuf=np.random.permutation(y_raw)
lr=LinearRegression().fit(X[:len(X)//2],y_shuf[:len(X)//2])
print(f'  Shuffled test R²: {lr.score(X[len(X)//2:],y_shuf[len(X)//2:]):.6f}')

# Shift returns by +1 bar (ret[t] vs latent[t-1])
n=len(X)
y_shifted=y_raw[1:]  # ret[t+1] vs latent[t]
lr2=LinearRegression().fit(X[:-1:2],y_shifted[::2])
print(f'  Shift+1 test R² (ret[t+1] vs latent[t]): {lr2.score(X[1:-1:2],y_shifted[1::2]):.6f}')

# Shift returns by -1 bar (ret[t-1] vs latent[t]) 
y_shifted2=y_raw[:-1]  # ret[t] vs latent[t+1]
lr3=LinearRegression().fit(X[1::2],y_shifted2[::2])
print(f'  Shift-1 test R² (ret[t] vs latent[t+1]): {lr3.score(X[2::2],y_shifted2[1::2]):.6f}')
