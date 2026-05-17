#!/usr/bin/env python3 -u
"""CPCV -> save OOS predictions (signal, vwap, dates) for all symbols."""
import sys, time, warnings, json
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from pipeline_cpcv import (load_binance, compute_features, cpcv_eval,
    SYMBOLS, FEATS, MB, OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True)

# Per-coin sigma from class-balance calibration
if (OUTPUT_DIR / 'coin_sigmas.json').exists():
    COIN_SIGMAS = json.load(open(OUTPUT_DIR / 'coin_sigmas.json'))
else:
    COIN_SIGMAS = {sym: 0.05 for sym in SYMBOLS}

print(f'Running CPCV + save OOS for {len(SYMBOLS)} symbols...')
t_start = time.time()

for sym in SYMBOLS:
    t0 = time.time()
    print(f'\n--- {sym} ---')

    df = load_binance(sym)
    df, _ = compute_features(df)

    df_c = df[FEATS].iloc[192:].reset_index(drop=True)
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

    # CPCV with VWAP labels
    sigma = COIN_SIGMAS.get(sym, 0.05)
    res = cpcv_eval(X, close_arr, ret1_arr, dates,
                    sigma=sigma, mb=MB,
                    n_blocks=6, n_est=40, depth=8, leaf=50, verbose=False,
                    vwap=vwap, vwap_ret_1=vwap_ret1)

    oos_sig = res['oos_sig']
    valid = res['oos_cnt'] > 0

    # Save signal + metadata
    np.save(OUTPUT_DIR / f'cpcv_oos_{sym}.npy', oos_sig[valid])
    if 'oos_pl' in res and 'oos_ps' in res:
        np.save(OUTPUT_DIR / f'cpcv_pl_{sym}.npy', res['oos_pl'][valid])
        np.save(OUTPUT_DIR / f'cpcv_ps_{sym}.npy', res['oos_ps'][valid])
    np.save(OUTPUT_DIR / f'cpcv_dates_{sym}.npy', dates[valid])
    np.save(OUTPUT_DIR / f'cpcv_vwap_{sym}.npy', vwap[valid])
    np.save(OUTPUT_DIR / f'cpcv_label_pnl_{sym}.npy', res['pnl'][valid])

    meta = {
        'signal_return_offset': int(res['signal_return_offset']),
        'entry_offset': int(res['entry_offset']),
        'mb': MB, 'sigma': sigma,
    }
    json.dump(meta, open(OUTPUT_DIR / f'cpcv_meta_{sym}.json', 'w'))

    sr = res['sr_by_th'][0.10]
    print(f'  Valid OOS: {valid.sum():,} bars')
    print(f'  SR@0.10:   {sr:.2f}')
    print(f'  Time:      {time.time()-t0:.0f}s')

print(f'\nTotal: {time.time()-t_start:.0f}s')
