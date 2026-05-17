# Crypto Research — 15m RF Trading Strategy

Multi-coin portfolio strategy using Random Forest + Triple Barrier labels on 15m Binance data.

## Architecture

```
src/
├── pipeline_cpcv.py      # Core: data loading, 62 features, TB labels, CPCV, PnL
├── run_cpcv_oos.py       # CPCV → save OOS predictions for all coins
├── backtest_oos.py       # Portfolio backtest from saved OOS predictions
├── batch_train.py        # Retrain final models (all coins)
├── online.py             # Online simulation (daily incremental)
├── feature_select_cpcv.py   # Feature selection via CPCV ablation
└── validate_feature_select.py  # Multi-coin validation
```

## Features (62 total)

| Group | Count | Features |
|-------|:-----:|----------|
| ret_* | 11 | Directional returns at 1/2/4/8/16/24/32/48/64/96/128 bars |
| abs_ret_* | 11 | Absolute returns (volatility regime) |
| ret_vol_corr_* | 4 | Return-volume correlation |
| v2: range/pos/frac | 9 | hl_range, close_pos, buy_frac (candle structure, buying pressure) |
| v2: vol_skew/ratio | 3 | Return skewness, volatility term structure |
| v3: trend quality | 4 | consec_up, consec_vol (streak, compression) |
| v3: price structure | 4 | ret_ma_dist, ret_range_pos (SMA distance, range position) |
| v3: volatility dyn | 4 | vol_delta, vol_max (vol acceleration, extremes) |
| v3: volume dyn | 4 | vol_surge, qv_surge (volume/quote surges) |
| v3: wick structure | 4 | up_wick, dn_wick (candle wick ratios) |
| v3: risk/autocorr | 4 | ret_sharpe, ret_acf (risk-adjusted momentum) |

## Current Results

- **Features:** 62 (all groups)
- **RF params:** n_est=40, max_depth=8, min_samples_leaf=50
- **TH:** per-coin SR-optimized (0.10-0.19)
- **EMA α:** 0.50
- **Fee:** 0.04%
- **Weight:** equal

| Metric | Value |
|--------|:-----:|
| Final equity | **$231K** (+2,210%) |
| SR (daily) | **5.32** |
| Max DD | -19.1% |
| Period | 4.49y |
| Annual return | +492% |

### History

```
6f5623d  Initial baseline (26 feats, TH=0.10)     $188K  SR=3.61
5731fd5  Restore per-coin TH (26 feats)           $229K  SR=3.93
198c6ed  11 v2 features (37 total)                 $252K  SR=4.91
4a96928  24 v3 features (62 total)                 $231K  SR=5.32  ← Current
```

## State Files

| File | Purpose |
|------|---------|
| `output/cpcv_oos_{sym}.npy` | CPCV OOS signals |
| `output/coin_sigmas.json` | Per-coin sigma from class-balance calibration |
| `output/online_state.json` | Daily online simulation state |
| `output/th_opt_37_sr.json` | TH optimization results |
| `model/*_final.pkl` | Trained Random Forest models |
