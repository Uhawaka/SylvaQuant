#!/usr/bin/env python3 -u
"""
Validate best feature subset (no_short_corr) across ALL 9 coins via CPCV + full portfolio backtest.
Compares: all_26 (baseline) vs no_short_corr (best found).
"""
import sys, warnings, time, json, os
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from pipeline_cpcv import (load_binance, compute_features, cpcv_eval,
    SYMBOLS, FEATS, MB, OUTPUT_DIR, backtest_pnl, TH_MAP)

# ── Feature subsets ──
SHORT_RET  = ['ret_1','ret_2','ret_4','ret_8']
MEDIUM_RET = ['ret_16','ret_24','ret_32','ret_48']
LONG_RET   = ['ret_64','ret_96','ret_128']
ALL_RET    = SHORT_RET + MEDIUM_RET + LONG_RET

ALL_ABS    = ['abs_ret_1','abs_ret_2','abs_ret_4','abs_ret_8',
              'abs_ret_16','abs_ret_24','abs_ret_32','abs_ret_48',
              'abs_ret_64','abs_ret_96','abs_ret_128']
CORR_FEATS = ['ret_vol_corr_16','ret_vol_corr_32','ret_vol_corr_64','ret_vol_corr_96']

# Best subset: remove ret_short + corr
BEST_FEATS = MEDIUM_RET + LONG_RET + ALL_ABS  # 18 features

# ── Config ──
TH = 0.10
FEE = 0.0004
EMA_ALPHA = 0.50


def run_coin(coin, feats, verbose=True):
    """Run CPCV for one coin with given features. Returns SR at TH=0.10."""
    df = load_binance(coin)
    df, _ = compute_features(df)

    avail = [f for f in feats if f in df.columns]
    df_c = df[avail].iloc[192:].reset_index(drop=True)
    close_arr = df['close'].to_numpy(np.float64)[192:]
    ret1_arr = df['ret_1'].to_numpy(np.float64)[192:]
    X = df_c.to_numpy(np.float32)
    dates = df['date'].iloc[192:].values

    vol = df['volume'].to_numpy(np.float64)[192:]
    qv = df['quote_vol'].to_numpy(np.float64)[192:]
    vwap = np.where(vol > 0, qv / vol, close_arr)
    vwap_log = np.log(np.maximum(vwap, 1e-10))
    vwap_ret1 = np.diff(vwap_log, prepend=vwap_log[0])

    sigma = 0.05
    t0 = time.time()
    res = cpcv_eval(X, close_arr, ret1_arr, dates,
                    sigma=sigma, mb=MB,
                    n_blocks=6, n_est=40, depth=8, leaf=50, verbose=False,
                    vwap=vwap, vwap_ret_1=vwap_ret1)
    t_elapsed = time.time() - t0

    valid = res['oos_cnt'] > 0
    oos_sig = res['oos_sig'][valid]
    oos_vwap = vwap[valid]
    sr_by_th = res['sr_by_th']

    # Save for full backtest
    sym_dir = OUTPUT_DIR / f'cpcv_{coin}'
    sym_dir.mkdir(exist_ok=True)
    np.save(sym_dir / f'cpcv_oos_{coin}.npy', oos_sig)
    np.save(sym_dir / f'cpcv_vwap_{coin}.npy', oos_vwap)
    np.save(sym_dir / f'cpcv_dates_{coin}.npy', dates[valid])
    if 'oos_pl' in res and 'oos_ps' in res:
        np.save(sym_dir / f'cpcv_pl_{coin}.npy', res['oos_pl'][valid])
        np.save(sym_dir / f'cpcv_ps_{coin}.npy', res['oos_ps'][valid])

    sr_10 = sr_by_th.get(TH, 0)
    n_trades_manual = int((oos_sig > TH).sum() + (oos_sig < -TH).sum())

    if verbose:
        print(f'  {coin:<10s} CPCV SR@0.10={sr_10:>+6.2f}  '
              f'n_sig={len(oos_sig):>8,d}  n_trades={n_trades_manual:>6,d}  [{t_elapsed:.0f}s]')

    return {
        'coin': coin,
        'sr': sr_10,
        'sig': oos_sig,
        'vwap': oos_vwap,
        'dates': dates[valid],
        'n_bars': valid.sum(),
        'n_trades': n_trades_manual,
    }


def run_full_cpcv(feats, label):
    """Run CPCV for all coins and save OOS predictions."""
    print(f'\n═══ CPCV: {label} ({len(feats)} feats) ═══')
    results = {}
    for sym in SYMBOLS:
        try:
            results[sym] = run_coin(sym, feats)
        except Exception as e:
            print(f'  {sym:<10s} ERROR: {e}')
            results[sym] = None
    return results


def backtest_portfolio(results, label):
    """Portfolio backtest from CPCV OOS predictions (same as backtest_oos.py)."""

    # Organize by date
    all_sig = {}; all_dates = {}; all_vwap = {}
    for sym, r in results.items():
        if r is None: continue
        all_sig[sym] = r['sig']
        all_dates[sym] = pd.to_datetime(r['dates'])
        all_vwap[sym] = r['vwap']

    if not all_sig:
        print(f'  No data for {label}')
        return

    common = sorted(set.intersection(*[set(all_dates[s]) for s in all_sig]))
    N = len(common)
    print(f'  {label}: Aligned {N:,} common bars')

    dl = {d: i for i, d in enumerate(common)}
    sig_m = np.zeros((N, len(SYMBOLS)))
    vwap_m = np.zeros((N, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        if sym not in all_sig: continue
        for k, dt in enumerate(all_dates[sym]):
            idx = dl.get(dt)
            if idx is not None:
                sig_m[idx, j] = all_sig[sym][k]
                vwap_m[idx, j] = all_vwap[sym][k]

    # Backtest each coin
    pnl_m = np.zeros((N, len(SYMBOLS)))
    pos_m = np.zeros((N, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        if sym not in all_sig: continue
        th_i = TH_MAP.get(sym, TH)
        pnl_m[:, j], pos_m[:, j], _ = backtest_pnl(
            sig_m[:, j], vwap_m[:, j], th_i, FEE, EMA_ALPHA)

    # Equal weight
    w = np.ones(len(SYMBOLS)) / len(SYMBOLS)
    port_pnl = (pnl_m * w.reshape(1, -1)).sum(axis=1)
    port_val = 10000 * np.cumprod(1 + port_pnl)

    # Stats
    n_bars = len(port_pnl)
    n_days = max(1, n_bars // 96)
    total_ret = port_val[-1] / 10000 - 1
    ann_ret = total_ret / n_days * 365
    ann_vol = np.std(port_pnl) * np.sqrt(96 * 365) if np.std(port_pnl) > 1e-15 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(port_val)
    max_dd = np.min((port_val - peak) / peak)

    # Daily SR
    from collections import defaultdict
    dp = defaultdict(float)
    for i in range(N):
        if port_pnl[i] != 0:
            dp[str(common[i])[:10]] += float(port_pnl[i])
    daily_vals = list(dp.values())
    daily_sr_val = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)
                    if len(daily_vals) > 5 and np.std(daily_vals) > 1e-10 else 0.0)

    print(f'  Period: {n_days/365:.2f}y  TH=per-coin  Fee={FEE}  Weight=equal  EMA={EMA_ALPHA}')
    print(f'  Ann Ret: {ann_ret:>+7.2%}')
    print(f'  SR (day): {daily_sr_val:>+7.2f}')
    print(f'  Max DD: {max_dd:>+7.2%}')
    print(f'  Final: ${port_val[-1]:>+,.2f}')

    # Per-coin detail
    print(f'\n{"Symbol":>10s} {"AnnRet":>8s} {"SR_day":>7s} {"Trades":>8s} {"MaxDD":>8s}')
    print('-' * 48)
    for j, sym in enumerate(SYMBOLS):
        if sym not in all_sig: continue
        r = pnl_m[:, j]
        ann_r = np.sum(r) / (n_days / 365)
        P = pos_m[:, j]
        trades = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], P]))) > 0))
        vj = 10000 * np.cumprod(1 + r)
        pk = np.maximum.accumulate(vj)
        dd_j = np.min((vj - pk) / pk)

        # Daily SR per coin
        dp2 = defaultdict(float)
        for i in range(N):
            if r[i] != 0:
                dp2[str(common[i])[:10]] += float(r[i])
        dv2 = list(dp2.values())
        sr_day_j = (np.mean(dv2) / np.std(dv2) * np.sqrt(252)
                    if len(dv2) > 5 and np.std(dv2) > 1e-10 else 0.0)
        print(f'{sym:>10s} {ann_r * 100:>+7.2f}% {sr_day_j:>+6.2f} {trades:>8,d} {dd_j:>+7.1%}')

    return {'sharpe': daily_sr_val, 'equity': port_val[-1], 'max_dd': max_dd}


if __name__ == '__main__':
    # 1) Run CPCV for best subset
    results_best = run_full_cpcv(BEST_FEATS, 'BEST (no_short_corr)')
    bt_best = backtest_portfolio(results_best, 'BEST')

    # 2) Run CPCV for baseline (all 26)
    results_all = run_full_cpcv(FEATS, 'ALL_26')
    bt_all = backtest_portfolio(results_all, 'ALL_26')

    # Summary
    print('\n' + '═' * 50)
    print('COMPARISON: ALL_26 vs BEST (no_short_corr)')
    print(f'{"Metric":<20s} {"ALL_26":>12s} {"BEST":>12s}')
    print('─' * 44)
    for k in ['sharpe', 'equity', 'max_dd']:
        v_all = bt_all.get(k, 0)
        v_best = bt_best.get(k, 0)
        if k == 'equity':
            print(f'{k:<20s} {v_all:>12,.0f} {v_best:>12,.0f}')
        else:
            print(f'{k:<20s} {v_all:>+11.2f} {v_best:>+11.2f}')
