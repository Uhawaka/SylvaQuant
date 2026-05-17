#!/usr/bin/env python3 -u
"""
CPCV-based feature selection on BTC, then validate on all coins.
Tests feature groups systematically, reports SR/PnL/DD for each subset.

Strategy: aggressive elimination — start from 26, drop harmful groups.
"""
import sys, warnings, time, json
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from pipeline_cpcv import (load_binance, compute_features, cpcv_eval,
    SYMBOLS, FEATS, MB, OUTPUT_DIR, backtest_pnl, print_results)

# ── Feature group definitions ──
SHORT_RET  = ['ret_1','ret_2','ret_4','ret_8']
MEDIUM_RET = ['ret_16','ret_24','ret_32','ret_48']
LONG_RET   = ['ret_64','ret_96','ret_128']
ALL_RET    = SHORT_RET + MEDIUM_RET + LONG_RET

SHORT_ABS  = ['abs_ret_1','abs_ret_2','abs_ret_4','abs_ret_8']
MEDIUM_ABS = ['abs_ret_16','abs_ret_24','abs_ret_32','abs_ret_48']
LONG_ABS   = ['abs_ret_64','abs_ret_96','abs_ret_128']
ALL_ABS    = SHORT_ABS + MEDIUM_ABS + LONG_ABS

CORR_FEATS = ['ret_vol_corr_16','ret_vol_corr_32','ret_vol_corr_64','ret_vol_corr_96']

# ── Subsets to test (aggressive: get rid of everything harmful) ──
SUBSETS = [
    ('01_all_26',        ALL_RET + ALL_ABS + CORR_FEATS),
    ('02_no_ret_short',  MEDIUM_RET + LONG_RET + ALL_ABS + CORR_FEATS),           # 22
    ('03_no_ret_all',    ALL_ABS + CORR_FEATS),                                    # 14
    ('04_only_abs_corr', ALL_ABS + CORR_FEATS),
    ('05_only_abs',      ALL_ABS),                                                  # 11
    ('06_abs_short',     SHORT_ABS),                                                # 4
    ('07_abs_short_mid', SHORT_ABS + MEDIUM_ABS),                                   # 8
    ('08_abs_no_corr',   ALL_ABS),                                                  # 11 (same as 05)
    ('09_no_corr',       ALL_RET + ALL_ABS),                                        # 22
    ('10_no_short_corr', MEDIUM_RET + LONG_RET + ALL_ABS),                          # 18
    ('11_only_short_abs',SHORT_ABS + MEDIUM_ABS + LONG_ABS[0:1]),                   # 9
    ('12_no_ret_mid',    SHORT_RET + LONG_RET + ALL_ABS + CORR_FEATS),             # 22
    ('13_abs_only_short_mid', SHORT_ABS + MEDIUM_ABS),                              # 8
]

TH = 0.10  # default threshold for SR reporting
FEE = 0.0004
EMA_ALPHA = 0.50


def run_single_cpcv(coin, feats, verbose=False):
    """Run CPCV + backtest for one coin with given feature subset."""
    df = load_binance(coin)
    # Use compute_features but it always computes ALL features
    # We filter to subset after compute
    df, all_feat_names = compute_features(df)

    # Check which requested features are available
    avail = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing and verbose:
        print(f'  Missing: {missing}')

    # Align features in the correct order
    df_c = df[avail].iloc[192:].reset_index(drop=True)
    close_arr = df['close'].to_numpy(np.float64)[192:]
    ret1_arr = df['ret_1'].to_numpy(np.float64)[192:]
    X = df_c.to_numpy(np.float32)
    dates = df['date'].iloc[192:].values

    # VWAP
    vol = df['volume'].to_numpy(np.float64)[192:]
    qv = df['quote_vol'].to_numpy(np.float64)[192:]
    vwap = np.where(vol > 0, qv / vol, close_arr)
    vwap_log = np.log(np.maximum(vwap, 1e-10))
    vwap_ret1 = np.diff(vwap_log, prepend=vwap_log[0])

    # CPCV
    sigma = 0.05  # default for BTC
    t0 = time.time()
    res = cpcv_eval(X, close_arr, ret1_arr, dates,
                    sigma=sigma, mb=MB,
                    n_blocks=6, n_est=40, depth=8, leaf=50, verbose=verbose,
                    vwap=vwap, vwap_ret_1=vwap_ret1)
    t_elapsed = time.time() - t0

    # Backtest on OOS predictions
    valid = res['oos_cnt'] > 0
    oos_sig = res['oos_sig'][valid]
    oos_vwap = vwap[valid]

    # Stats
    total_n = valid.sum()
    pnl, pos, trades = backtest_pnl(oos_sig, oos_vwap, TH, FEE, EMA_ALPHA)
    port_val = 10000 * np.cumprod(1 + pnl)
    n_bars = len(pnl)
    n_days = max(1, n_bars // 96)
    total_ret = port_val[-1] / 10000 - 1
    ann_ret = total_ret / n_days * 365
    ann_vol = np.std(pnl) * np.sqrt(96 * 365) if np.std(pnl) > 1e-15 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(port_val)
    max_dd = np.min((port_val - peak) / peak)

    # CPCV SR (from _eval_thresholds)
    cpcv_sr = res['sr_by_th'].get(TH, 0)

    # Feature importance
    fi = res.get('feat_imp', np.zeros(len(avail)))
    top3_idx = np.argsort(fi)[::-1][:3]
    top3 = [(avail[i], fi[i]) for i in top3_idx]

    return {
        'feats': len(avail),
        'n': total_n,
        'cpcv_sr': cpcv_sr,
        'backtest_ret': total_ret,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'top3': top3,
        'time': t_elapsed,
        'port_val': port_val[-1],
        'feat_imp': fi,
        'feat_names': avail,
    }


def run_all_subsets(coin='BTCUSDT'):
    """Run CPCV for all subsets and print comparison table."""
    results = {}
    print(f'═══ CPCV Feature Selection: {coin} ═══')
    print(f'  TH={TH}, Fee={FEE}, EMA_alpha={EMA_ALPHA}, mb={MB}')
    print(f'  {len(SUBSETS)} subsets to test\n')

    for name, feats in SUBSETS:
        print(f'  ── {name} ({len(feats)} feats) ──')
        try:
            r = run_single_cpcv(coin, feats, verbose=False)
            results[name] = r
            print(f'    CPCV SR={r["cpcv_sr"]:.2f} | Backtest: ${r["port_val"]:>+,.0f} '
                  f'SR={r["sharpe"]:.2f} DD={r["max_dd"]:.1%} '
                  f'Trades={r["trades"]:,} [{r["time"]:.0f}s]')
            if r['top3']:
                print(f'    Top3: {", ".join(f"{f}={v:.3f}" for f,v in r["top3"])}')
        except Exception as e:
            print(f'    ERROR: {e}')
            results[name] = None

    # Print comparison table
    print('\n' + '═' * 80)
    print(f'{"Subset":<24s} {"N_feat":>6s} {"CPCV_SR":>8s} {"BT_SR":>6s} '
          f'{"AnnRet":>8s} {"MaxDD":>7s} {"Trades":>8s} {"Equity":>10s}')
    print('─' * 80)
    for name, _ in SUBSETS:
        r = results.get(name)
        if r is None:
            print(f'{name:<24s} {"ERROR":>6s}')
        else:
            ann_ret_r = r['ann_ret'] * 100
            print(f'{name:<24s} {r["feats"]:>6d} {r["cpcv_sr"]:>+8.2f} '
                  f'{r["sharpe"]:>+6.2f} {ann_ret_r:>+7.1f}% '
                  f'{r["max_dd"]:>+6.1%} {r["trades"]:>8,d} {r["port_val"]:>+10,.0f}')

    # Find best
    best_name = max(results, key=lambda x: results[x]['sharpe'] if results[x] else -999)
    best_r = results[best_name]
    print('\n' + '═' * 80)
    print(f'🏆 Best: {best_name} ({best_r["feats"]} feats)')
    print(f'   CPCV SR={best_r["cpcv_sr"]:.2f} | BT SR={best_r["sharpe"]:.2f} '
          f'| AnnRet={best_r["ann_ret"]:.1%} | DD={best_r["max_dd"]:.1%} '
          f'| $={best_r["port_val"]:,.0f}')
    print()  # flush

    return results, best_name


if __name__ == '__main__':
    results, best_name = run_all_subsets('BTCUSDT')

    # Save results
    save = {}
    for name, r in results.items():
        if r:
            save[name] = {k: v for k, v in r.items() if k not in ('feat_imp', 'feat_names')}
    json.dump(save, open(OUTPUT_DIR / 'feature_select_results.json', 'w'), indent=2, default=str)
    print(f'Results saved to {OUTPUT_DIR / "feature_select_results.json"}')
