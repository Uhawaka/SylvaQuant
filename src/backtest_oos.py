#!/usr/bin/env python3 -u
"""Backtest using CPCV OOS predictions. Same PnL code as online.py."""
import warnings, json, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_cpcv import backtest_pnl, SYMBOLS, OUTPUT_DIR, FEATS

TH = float(os.getenv('TH', '0.10'))
FEE = float(os.getenv('FEE', '0.0004'))
EMA_ALPHA = float(os.getenv('EMA_ALPHA', '0.50'))
WEIGHT_SCHEME = os.getenv('WEIGHT_SCHEME', 'equal').lower()
TH_PER_COIN = os.getenv('TH_PER_COIN', '0') in ('1', 'true', 'True')
PROB_MODE = os.getenv('PROB_MODE', '0') in ('1', 'true', 'True')
PTH = float(os.getenv('PTH', '0.55'))
TH_MAP = {
    'BTCUSDT': 0.25, 'ETHUSDT': 0.25, 'SOLUSDT': 0.25, 'BNBUSDT': 0.25,
    'ADAUSDT': 0.30, 'XRPUSDT': 0.30, 'DOGEUSDT': 0.30, 'DOTUSDT': 0.30, 'AVAXUSDT': 0.30,
}


def _pnl_from_probs(pl, ps, vwaps, pth, fee, ema_alpha=None):
    """Prob-based PnL. If ema_alpha set, smooths prob-difference signal."""
    sig = pl - ps
    if ema_alpha is not None:
        s = np.asarray(sig, np.float64).copy()
        for t in range(1, len(s)):
            s[t] = ema_alpha * s[t] + (1 - ema_alpha) * s[t-1]
        sig = s
    n = len(sig)
    P = np.zeros(n)
    P[2:] = np.where(sig[:-2] > pth, 1.0, np.where(sig[:-2] < -pth, -1.0, 0.0))
    R = np.zeros(n)
    R[1:] = vwaps[1:] / vwaps[:-1] - 1
    dP = np.abs(np.diff(P))
    dP = np.concatenate([[0.0], dP])
    pnl = P * R - dP * fee
    trades = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], P]))) > 0))
    return pnl, P, trades


# ── Load CPCV OOS ──
print('Loading CPCV OOS Predictions')
all_sig = {}; all_dates = {}; all_vwap = {}
all_pl = {}; all_ps = {}
for sym in SYMBOLS:
    p = OUTPUT_DIR / f'cpcv_oos_{sym}.npy'
    if not p.exists():
        continue
    all_sig[sym] = np.load(p)
    all_dates[sym] = pd.to_datetime(np.load(OUTPUT_DIR / f'cpcv_dates_{sym}.npy'))
    all_vwap[sym] = np.load(OUTPUT_DIR / f'cpcv_vwap_{sym}.npy')
    if PROB_MODE:
        plp = OUTPUT_DIR / f'cpcv_pl_{sym}.npy'
        psp = OUTPUT_DIR / f'cpcv_ps_{sym}.npy'
        if plp.exists() and psp.exists():
            all_pl[sym] = np.load(plp)
            all_ps[sym] = np.load(psp)

# Align to common index
common = sorted(set.intersection(*[set(all_dates[sym]) for sym in SYMBOLS if sym in all_dates]))
N = len(common)
print(f'Aligned: {N:,} common bars\n')

dl = {d: i for i, d in enumerate(common)}
sig_m = np.zeros((N, len(SYMBOLS)))
vwap_m = np.zeros((N, len(SYMBOLS)))
pl_m = np.zeros((N, len(SYMBOLS))) if PROB_MODE else None
ps_m = np.zeros((N, len(SYMBOLS))) if PROB_MODE else None
for j, sym in enumerate(SYMBOLS):
    if sym not in all_sig:
        continue
    for k, dt in enumerate(all_dates[sym]):
        idx = dl.get(dt)
        if idx is not None:
            sig_m[idx, j] = all_sig[sym][k]
            vwap_m[idx, j] = all_vwap[sym][k]
            if PROB_MODE and sym in all_pl and sym in all_ps:
                pl_m[idx, j] = all_pl[sym][k]
                ps_m[idx, j] = all_ps[sym][k]

# ── Backtest ──
pnl_m = np.zeros((N, len(SYMBOLS)))
pos_m = np.zeros((N, len(SYMBOLS)))
for j, sym in enumerate(SYMBOLS):
    if sym not in all_sig:
        continue
    if PROB_MODE and pl_m is not None and sym in all_pl and sym in all_ps:
        pnl_m[:, j], pos_m[:, j], _ = _pnl_from_probs(
            pl_m[:, j], ps_m[:, j], vwap_m[:, j], PTH, FEE, ema_alpha=EMA_ALPHA)
    else:
        th_i = TH_MAP.get(sym, TH) if TH_PER_COIN else TH
        pnl_m[:, j], pos_m[:, j], _ = backtest_pnl(
            sig_m[:, j], vwap_m[:, j], th_i, FEE, ema_alpha=EMA_ALPHA)

# Weighting
if WEIGHT_SCHEME == 'invvolcap':
    vol = pnl_m.std(axis=0)
    vol = np.where(vol < 1e-12, np.nan, vol)
    w = 1.0 / vol
    w = np.nan_to_num(w, nan=0.0)
    if w.sum() > 0:
        w = w / w.sum()
        cap = 0.20
        for _ in range(10):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if under.any() and excess > 0 and w[under].sum() > 0:
                w[under] += excess * (w[under] / w[under].sum())
        w = w / w.sum() if w.sum() > 0 else w
    weights = w
else:
    weights = np.ones(len(SYMBOLS), dtype=float)
    weights = weights / weights.sum()

port_pnl = (pnl_m * weights.reshape(1, -1)).sum(axis=1)
port_val = 10000 * np.cumprod(1 + port_pnl)

# ── Stats ──
n_days = max(1, N // 96)
total_ret = port_val[-1] / 10000 - 1
ann_ret = total_ret / n_days * 365
ann_vol = np.std(port_pnl) * np.sqrt(96 * 365) if np.std(port_pnl) > 1e-15 else 0.0
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
peak = np.maximum.accumulate(port_val)
max_dd = np.min((port_val - peak) / peak)


def daily_sr(pnl_series):
    dp = defaultdict(float)
    for i in range(N):
        if pnl_series[i] != 0:
            dp[str(common[i])[:10]] += float(pnl_series[i])
    vals = list(dp.values())
    if len(vals) > 5 and np.std(vals) > 1e-10:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0.0


# ── Print ──
th_desc = f'prob@{PTH}' if PROB_MODE else ('per-coin' if TH_PER_COIN else f'{TH}')
print(f'══ PORTFOLIO RESULTS ══')
print(f'  Period:   {n_days/365:.2f}y  TH={th_desc}  Fee={FEE}  Weight={WEIGHT_SCHEME}  EMA_alpha={EMA_ALPHA}')
print(f'  Ann Ret:  {ann_ret:>+7.2%}')
print(f'  Ann Vol:  {ann_vol:>+7.2%}')
print(f'  SR (bar): {sharpe:>+7.2f}')
print(f'  SR (day): {daily_sr(port_pnl):>+7.2f}')
print(f'  Max DD:   {max_dd:>+7.2%}')
print(f'  Final:    ${port_val[-1]:>+,.2f}')
print()
print(f'{"Symbol":>10s} {"AnnRet":>8s} {"SR_day":>7s} {"SR_bar":>7s} {"MaxDD":>8s} {"Trades":>8s} {"Win%":>6s} {"AvgBP":>7s}')
print('-' * 68)
for j, sym in enumerate(SYMBOLS):
    if sym not in all_sig:
        continue
    r = pnl_m[:, j]
    ann_r = np.sum(r) / (n_days / 365)
    sr_bar_j = float(np.mean(r) / np.std(r) * np.sqrt(96 * 365)) if np.std(r) > 1e-15 else 0.0
    P = pos_m[:, j]
    trades = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], P]))) > 0))
    wins = np.sum(r[P != 0] > 0)
    tot = np.sum(P != 0)
    winrate = wins / tot * 100 if tot > 0 else 0
    avg = np.mean(r[P != 0]) * 10000 if tot > 0 else 0
    vj = 10000 * np.cumprod(1 + r)
    pk = np.maximum.accumulate(vj)
    dd_j = np.min((vj - pk) / pk)
    print(f'{sym:>10s} {ann_r * 100:>+7.2f}% {daily_sr(r):>+6.2f} {sr_bar_j:>+7.2f} {dd_j:>+7.2%} {trades:>8,d} {winrate:>5.1f}% {avg:>+7.1f}')

# ── Save ──
np.savetxt(OUTPUT_DIR / 'backtest_oos_equity.csv',
           np.column_stack([common, port_val, port_pnl]),
           delimiter=',', fmt='%s,%.6f,%.8f', header='date,equity,return', comments='')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 6))
dt_arr = [pd.Timestamp(d) for d in common]
ax.plot(dt_arr, port_val, linewidth=0.8, color='navy')
ax.set_title(f'Portfolio Backtest (TH={TH}, fee={FEE}, SR_day={daily_sr(port_pnl):.2f})')
ax.set_ylabel('Equity ($)')
ax.axhline(10000, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'backtest_oos_equity.png', dpi=120)
print(f'\nDone.')
