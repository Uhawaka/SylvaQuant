# SylvaQuant — RL Dynamic Weight Allocation (Experiment)

## Overview

This sub-project explores **reinforcement learning for portfolio weight allocation** on a 9-coin cryptocurrency portfolio (15m bars). The approach uses:

1. **AE** (Autoencoder): Compress 558 features (62×9 coins) → 16-dim market latent state
2. **CFM** (Conditional Flow Matching): Generate synthetic (latent, return) pairs matching the real distribution
3. **GRPO** (Group Relative Policy Optimization): Train a policy in a synthetic "dream world" to output optimal portfolio weights

## Pipeline

```
Raw Features (558) → AE → Latent (16) 
                                ↓
                    CFM → Synthetic (latent, return) pairs
                                ↓
                    GRPO → Policy → weights ∈ [-1, 1]^9
                                ↓
                    Backtest (VWAP fill, offset=2, fee=0.04%)
```

## Files

| File | Purpose |
|------|---------|
| `src/train_market_latent.py` | Train AE: 558→16 market latent |
| `src/train_rl_policy.py` | GRPO training on synthetic CFM data |
| `src/backtest_rl.py` | Time-series backtest with fees |
| `RL_ARCH.md` | This document |
| `data/market_latent.npy` | 189K × 16 latents |
| `data/market_latent_ae.pt` | Trained AE model |
| `data/synthetic_cfm.npz` | 200K synthetic (latent, return) pairs |
| `data/cfm_joint.pt` | Trained CFM model |
| `data/rl_policy.pt` | Trained policy checkpoint |

## Training

### Stage 1: Market Latent (AE)
```bash
python src/train_market_latent.py
```
AE compresses 558-dim feature space → 16-dim latent. Dumbbell architecture (SiLU, ResBlock, dropout=0.1). 20 epochs.

### Stage 2: CFM (Synthetic Data Generation)
```bash
python src/train_cfm.py
```
Conditional Flow Matching (Lipman et al., 2022) on joint [latent(16), return(9)] distribution. Time-conditioned MLP with sinusoidal embedding. Generates 200K synthetic pairs.

### Stage 3: GRPO (Policy Training)
```bash
python src/train_rl_policy.py
```
Policy: tanh-Gaussian → weights ∈ [-1, 1]. 
- K=32 samples per state (group competition)
- Turnover-based fee in reward
- Group-normalized advantage (GRPO)
- Entropy + KL regularization

### Backtest
```bash
python src/backtest_rl.py
```
VWAP fill at offset=2. Fee 0.04% on turnover.

## Results

| Metric | Value |
|--------|-------|
| Pre-fee SR | 3-5 (config dependent) |
| After-fee SR | ~0 (fee eats all alpha) |
| Avg turnover | 2.4 (raw) |
| Avg Σ|weight| | 8.6 |

## Key Findings

1. **CFM works well**: Generates realistic (latent, return) pairs from noise. Synthetic return std ≈ 0.0052 (target 0.0071, ratio 73%).

2. **GRPO + synthetic data works**: Policy trained purely on synthetic data transfers to real data (no leakage).

3. **Fee is the killer**: At 0.04% per trade, turnover of 2.4/bar consumes ~0.1% of capital per bar. The signal (pre-fee SR≈4) isn't strong enough to overcome this friction.

4. **L1 normalization hurts**: Σ|w| fixed at 1 reduces returns 8.6x, but turnover fee only drops 0.5x. Net becomes even worse.

## Open Questions

- Can a lower fee environment (0.01% maker) make this viable?
- Would higher-timeframe returns (1h/4h) with proportionally lower turnover help?
- Could threshold-based trading (only when signal confidence > 0.5) reduce turnover enough?

## References

- Lipman et al., "Flow Matching for Generative Modeling" (2022)
- DeepSeekMath, "GRPO: Group Relative Policy Optimization" (2024)
- Haarnoja et al., "Soft Actor-Critic" (2018)
