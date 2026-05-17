#!/usr/bin/env python3 -u
"""Batch train all 9 models with per-coin class-balanced sigma, VWAP labels."""
import pickle, time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
from pipeline_cpcv import (load_binance, compute_features, tb_labels,
    SYMBOLS, FEATS, MB, RF_N_EST, RF_DEPTH, RF_LEAF,
    OUTPUT_DIR, MODEL_DIR)

MODEL_DIR.mkdir(exist_ok=True)
COIN_SIGMAS = json.load(open(OUTPUT_DIR / 'coin_sigmas.json'))
print(f'Batch training {len(SYMBOLS)} symbols: mb={MB}, VWAP labels')
print(f'Per-coin sigmas: {COIN_SIGMAS}')
print('=' * 60)

for sym in SYMBOLS:
    t0 = time.time()
    SIGMA = COIN_SIGMAS[sym]
    print(f'\n--- {sym} (sigma={SIGMA}) ---')

    df = load_binance(sym)
    df = compute_features(df)[0]
    avail = [c for c in FEATS if c in df.columns]

    # VWAP
    vwap = np.where(df['volume'] > 0, df['quote_vol'] / df['volume'], df['close']).astype(np.float64)
    df_c = df[avail].iloc[192:].reset_index(drop=True)
    X_arr = df_c.to_numpy(np.float32)
    vwap_arr = vwap[192:]
    n = len(X_arr)

    vwap_log = np.log(np.maximum(vwap_arr, 1e-10))
    vwap_ret1 = np.diff(vwap_log, prepend=vwap_log[0])
    train_vol = float(vwap_ret1[:int(n * 0.70)].std() * np.sqrt(MB))
    label, _, _ = tb_labels(vwap_arr, train_vol, SIGMA, SIGMA, MB, entry_offset=1)

    lp = np.mean(label > 0) * 100
    sp = np.mean(label < 0) * 100
    fp = np.mean(label == 0) * 100
    print(f'  {n:,} bars, vol={train_vol*100:.3f}%, barrier={SIGMA*train_vol*100:.3f}%, L={lp:.1f}% S={sp:.1f}% F={fp:.1f}%')

    if len(label) < 10:
        print('  SKIP: too few bars')
        continue

    rf = RandomForestClassifier(
        n_estimators=RF_N_EST, max_depth=RF_DEPTH, min_samples_leaf=RF_LEAF,
        class_weight='balanced', n_jobs=-1, random_state=42,
    )
    rf.fit(X_arr, label.astype(int))

    save_dict = {
        'model': rf, 'feat': avail, 'train_vol': float(train_vol),
        'sigma': SIGMA, 'mb': MB, 'symbol': sym,
        'train_date': str(pd.Timestamp.utcnow()),
    }
    path = MODEL_DIR / f'{sym.lower()}_final.pkl'
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f'  Saved: {path.name}, imp_top={rf.feature_importances_.max():.3f}, {time.time()-t0:.0f}s')

print(f'\n{"=" * 60}\nAll done.')
