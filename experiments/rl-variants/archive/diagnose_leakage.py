#!/usr/bin/env python3 -u
"""Diagnose data leakage in RL experiment pipeline."""
import sys,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))

d=np.load(str(ROOT/'data/rl_exp/exp_data.npz'))
lat=d['latents'];ret=d['raw_ret'];sig=d['sig_pnl']
tr,va,te=d['train_idx'],d['val_idx'],d['test_idx']
print('═══ Leakage Diagnosis ═══\n')
print(f'latents: {lat.shape}  returns: {ret.shape}')
print(f'train: {tr}  val: {va}  test: {te}')

# ── 1. Check alignment: do latents[t] predict ret[t]? ──
print('\n── 1. Latent → return predictability (linear) ──')
for name,sl in [('Train',tr),('Val',va),('Test',te)]:
    n=sl[1]-sl[0];mid=n//2
    X=lat[sl[0]:sl[1]];y=ret[sl[0]:sl[1]].mean(1)
    lr=LinearRegression().fit(X[:mid],y[:mid])
    r2=lr.score(X[mid:],y[mid:])
    print(f'  {name}: n={n} R²={r2:.6f} (should be ~0 for no leakage)')

# ── 2. VWAP alignment sanity check ──
print('\n── 2. VWAP/dates alignment check ──')
dates=np.array([str(d) for d in np.load(ROOT/'data/market_latent_dates.npy',allow_pickle=True)])
# Check: is latent[t] from bar t, reward[t] from 2 bars ahead?
# lat_align = latents[:-2], ret[t] = VWAP[t+2]/VWAP[t+1]-1
# So latent_aligned[t] should be from an earlier timestamp than ret[t]
lat_align=lat  # already aligned as lat[:-2]
print(f'  Sample timestamps:')
for i in range(5):
    print(f'    latent[{i}] ret: bar t   reward: VWAP[t+2]/VWAP[t+1]-1')
    print(f'    dates: {dates[i]}  vs  VWAP dates...')

# ── 3. Check latent normalization ──
print('\n── 3. Latent normalization ──')
lat_full=np.load(str(ROOT/'data/market_latent.npy'))
print(f'  Full latents: {lat_full.shape}')
print(f'  Aligned (lat[:-2]): {lat.shape}')
mu=lat_full.mean(0);sd=lat_full.std(0).clip(1e-6)
# Check if lat is pre-normalized
if np.abs(lat_full.mean())<1 and np.abs(lat_full.std()-1)<1:
    print(f'  ⚠️ Latents appear pre-normalized with full-dataset stats')
    print(f'     mean={lat_full.mean():.4f} std={lat_full.std():.4f}')
    print(f'     → look-ahead bias: bar t uses mu/sd from all bars including future')

# ── 4. Check: does the VWAP[t] info actually leak? ──
# Test: linear model using only volume[t] should predict VWAP ret[t]
print('\n── 4. Feature → VWAP[t] predictability ──')
# Check if feature columns that encode volume/quote_vol correlate with VWAP return
# raw_ret[t] = VWAP[t+2]/VWAP[t+1]-1
# Does latent[t] contain info about VWAP[t+2]?
# Quick test: autocorrelation of VWAP returns
for j in range(3):
    r=ret[:5000,j]
    ac=np.corrcoef(r[:-2],r[2:])[0,1]
    print(f'  coin[{j}] VWAP ret lag-2 autocorr: {ac:.4f}')
    # If lag-2 autocorr is non-zero, then ret[t] can be predicted from ret[t-2]
    # But ret[t-2] is NOT in latent[t] (latent[t] has features from bar t only)

# ── 5. Check: is the issue specific to the exp data, or also in stable training? ──
print('\n── 5. Compare with stable train_rl_policy.py data ──')
# The stable version also gets SR≈3-4, what's different?
# Key difference: stable version uses CFM synthetic returns scaled to match real vol
# Not: both use the same latents and VWAP alignment

# ── 6. Direct test: does a simple model overfit? ──
print('\n── 6. Single-feature regression test ──')
from sklearn.linear_model import Ridge
X_train=lat[tr[0]:tr[1]];y_train=ret[tr[0]:tr[1]].mean(1)
X_test=lat[te[0]:te[1]];y_test=ret[te[0]:te[1]].mean(1)
for a in [0,1e-3,1]:
    r=Ridge(alpha=a).fit(X_train,y_train)
    pred=r.predict(X_test)
    # Sharpe of predictions (as a trading signal)
    sig=np.where(pred>0,1,-1)
    sr=sig[:len(pred)-2].mean()/max(sig[:len(pred)-2].std(),1e-8)*np.sqrt(24*365)
    print(f'  Ridge(α={a}): test Sharpe={sr:.2f} (should be ~0)')

# ── 7. The key check: latent[t] vs reward[t] timing ──
print('\n── 7. Final timing verification ──')
print(f'  lat_align[t] = latents[t] (bar t features)')
print(f'  ret[t] = VWAP[t+2]/VWAP[t+1]-1')
print(f'  → If no leakage, a linear model on lat_align[t] should NOT predict ret[t]')
# This is already checked in #1

# ── 8. Check: does the feature set include VWAP or closely related quantities? ──
print('\n── 8. Feature list check ──')
try:
    from pipeline_cpcv import FEATS
    print(f'  Feature count: {len(FEATS)}')
    vwap_related=[f for f in FEATS if any(k in f.lower() for k in ['vwap','volume','quote','amount','taker'])]
    print(f'  VWAP/volume related features: {len(vwap_related)}')
    for f in vwap_related[:10]:
        print(f'    {f}')
except Exception as e:
    print(f'  Error: {e}')
