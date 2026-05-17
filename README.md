# Crypto Research — 15m RF Trading Strategy

Multi-coin portfolio strategy using Random Forest + Triple Barrier labels on 15m Binance data.

## Architecture

```
src/
├── pipeline_cpcv.py      # Core: data loading, features, TB labels, CPCV, PnL
├── run_cpcv_oos.py       # CPCV → save OOS predictions for all coins
├── backtest_oos.py       # Portfolio backtest from saved OOS predictions
├── batch_train.py        # Retrain final models (all coins)
├── online.py             # Online simulation (daily incremental)
├── feature_select_cpcv.py   # Feature selection via CPCV ablation
└── validate_feature_select.py  # Multi-coin validation
```

## Current Results (baseline)

- **All 26 ret features, TH=0.10, EMA=0.50, equal weight**
- Final equity: **$188,117** (+1,781%), SR_day=3.61, DD=-23%
- Per-coin TH optimized: **$234K**, SR_day=4.21, DD=-17%

## State Files

| File | Purpose |
|------|---------|
| `output/cpcv_oos_{sym}.npy` | CPCV OOS signals |
| `output/coin_sigmas.json` | Per-coin sigma from class-balance calibration |
| `output/online_state.json` | Daily online simulation state |
| `model/*_final.pkl` | Trained Random Forest models |
