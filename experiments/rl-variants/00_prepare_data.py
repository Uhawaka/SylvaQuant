#!/usr/bin/env python3 -u
"""Build aligned dataset for RL experiments: market_latent + VWAP returns."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','ADAUSDT','XRPUSDT','DOGEUSDT','DOTUSDT','AVAXUSDT']
OUT = ROOT / 'data' / 'rl_exp'
OUT.mkdir(parents=True, exist_ok=True)

print('═══ Prepare RL Experiment Data ═══\n')

# ── Load market latent + dates ──
latents = np.load(ROOT / 'data' / 'market_latent.npy').astype(np.float32)
dates = np.load(ROOT / 'data' / 'market_latent_dates.npy', allow_pickle=True)
N = len(latents)
print(f'Market latents: {latents.shape} (N={N:,})')
print(f'Date range: {dates[0]} → {dates[-1]}')

# ── Load OOS vwap + dates per coin, align to market_latent ──
vwap_aligned = np.zeros((N, len(SYMBOLS)), np.float32)
oos_aligned = np.zeros((N, len(SYMBOLS)), np.float32)
date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)}

for j, sym in enumerate(SYMBOLS):
    vwap = np.load(ROOT / 'output' / f'cpcv_vwap_{sym}.npy')
    oos = np.load(ROOT / 'output' / f'cpcv_oos_{sym}.npy')
    sym_dates = np.load(ROOT / 'output' / f'cpcv_dates_{sym}.npy')
    # Align
    count = 0
    for k, dt in enumerate(sym_dates):
        idx = date_index.get(pd.Timestamp(dt))
        if idx is not None:
            vwap_aligned[idx, j] = np.float32(vwap[k])
            oos_aligned[idx, j] = np.float32(oos[k])
            count += 1
    print(f'  {sym}: {count}/{N} bars aligned (len={len(vwap):,})')

# ── Compute VWAP returns (signal_return_offset=2) ──
# return[t] = vwap[t+2] / vwap[t+1] - 1
# Signal at bar t → position at VWAP[t+1] (entry_offset=1) → exit at VWAP[t+2]
# Reward for action at time t = return[t] = VWAP[t+2]/VWAP[t+1]-1
# latents[t] is from bar t — does NOT see bar t+2 data (no look-ahead)
Nret = N - 2
raw_ret = np.zeros((Nret, len(SYMBOLS)), np.float32)
for j in range(len(SYMBOLS)):
    r = vwap_aligned[2:, j] / vwap_aligned[1:-1, j] - 1.0
    raw_ret[:, j] = np.where(np.isfinite(r), r, 0.0)
raw_ret = np.clip(raw_ret, -0.05, 0.05).astype(np.float32)
print(f'  Return timing: reward[t] = VWAP[t+2]/VWAP[t+1]-1 (offset=2, {Nret} bars)')

# ── Signal-derived PnL (matches backtest_pnl offset=2 logic) ──
# position[t] = signal[t] (entered at VWAP[t+1]), return at t = VWAP[t+2]/VWAP[t+1]-1
sig_pnl = np.zeros((Nret, len(SYMBOLS)), np.float32)
for j in range(len(SYMBOLS)):
    sig = oos_aligned[:-2, j]
    pos = np.where(sig > 0.10, 1.0, np.where(sig < -0.10, -1.0, 0.0))
    sig_pnl[:, j] = pos * raw_ret[:, j]

# ── Train/val/test split (time-based) ──
# Align latents with returns (offset=2):
#   lat_align[t] = latents[t] (features from bar t)
#   raw_ret[t]   = VWAP[t+2]/VWAP[t+1]-1 (return entered at bar t+1, exited at bar t+2)
#   → drop last 2 bars to match: latents[:-2] aligns with ret
lat_align = latents[:-2]   # (N-2) × 16 — features from bar t
dates_align = dates[:-2]   # (N-2)
N_align = len(lat_align)

n_train = int(N_align * 0.7)
n_val = int(N_align * 0.15)
idx_train = slice(0, n_train)
idx_val = slice(n_train, n_train + n_val)
idx_test = slice(n_train + n_val, N_align)

print(f'\nSplits: train={n_train:,}  val={n_val:,}  test={N_align-n_train-n_val:,}')

# ── Save ──
np.savez(OUT / 'exp_data.npz',
    latents=lat_align,
    dates=np.array([str(d) for d in dates_align]),
    raw_ret=raw_ret,
    sig_pnl=sig_pnl,
    oos_sig=oos_aligned[:-2],
    train_idx=np.array([0, n_train]),
    val_idx=np.array([n_train, n_train + n_val]),
    test_idx=np.array([n_train + n_val, N_align]))

print(f'\n✅ Saved to {OUT / "exp_data.npz"}')
print(f'   latents:    {lat_align.shape}')
print(f'   raw_ret:    {raw_ret.shape}')
print(f'   sig_pnl:    {sig_pnl.shape}')
print(f'   raw_ret mean/std: {raw_ret.mean():.6f}/{raw_ret.std():.6f}')
print(f'   sig_pnl mean/std: {sig_pnl.mean():.6f}/{sig_pnl.std():.6f}')

# Quick baseline: equal-weight Sharpe (with correct offset=2 timing)
for name, sl in [('Train', idx_train), ('Val', idx_val), ('Test', idx_test)]:
    ew_ret = raw_ret[sl].mean(axis=1)
    sr = ew_ret.mean() / (ew_ret.std() + 1e-8) * np.sqrt(24*365)
    print(f'  EW Sharpe ({name}): {sr:.4f}')
