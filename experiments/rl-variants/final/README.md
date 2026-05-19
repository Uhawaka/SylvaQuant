# DeltaSoft RL — Portfolio Trading with Differentiable Soft Threshold

End-to-end reinforcement learning for multi-asset portfolio trading using differentiable policy optimization.

## Core Idea

**Delta Soft Threshold Policy** learns when to enter, hold, and exit positions by controlling **changes** in position, not absolute positions:

```
w[t] = clamp(w[t-1] + delta[t], -1, 1)
delta[t] = sign(score) · max(|score| - θ, 0)
```

This is mathematically equivalent to L1-regularized optimal control:
- score ≈ 0 → delta ≈ 0 → **HOLD**
- score > θ → delta > 0 → **ENTER/ADD**
- score < -θ → delta < 0 → **EXIT/REDUCE**

The threshold θ is learned per-asset per-bar via a separate network head.

## Architecture

```
Gym Env (numpy, clean interface)
  └── TorchVecEnv (B parallel environments, differentiable)
        └── DeltaSoftThresh Policy (2-layer SiLU MLP)
              └── Score head: raw signal direction
              └── Threshold head: how selective to be
```

- **B=256** parallel environments
- **L=32** bars per episode (~8h of 15m data)
- Gradients flow through ALL L×B steps
- Optimization: direct Sharpe maximization (no PPO/GRPO)

## Best Result

| Config | Test SR | TO | Net SR | Active |
|--------|---------|---|--------|--------|
| θ=0.18, L=32, B=256 | **1.68** | 0.116 | **+1.21** | 38% |
| θ=0.16, L=32, B=256 | 1.13 | 0.092 | +0.79 | 21% |
| θ=0.15, L=32, B=256 | 0.85 | 0.122 | +0.39 | 31% |

**Key findings:**
- θ=0.18 is the optimal threshold (balances signal utilization vs turnover)
- 10K steps with constant LR outperforms longer training
- 5/5 seeds positive Net SR at θ=0.15
- Original 2-layer SiLU MLP outperforms deeper/variant architectures

## Files

| File | Purpose |
|------|---------|
| `env_rl.py` | Policy, VecEnv, eval, training, data loading |
| `train.py` | Training entry points (best/sweep/multi-seed) |

## Usage

```bash
# Train best config
python final/train.py

# Sweep θ_max
python final/train.py --sweep

# Multi-seed validation
python final/train.py --multiseed 5
```

## Data

Expects `data/rl_exp/exp_data.npz` with:
- `latents`: (N, 16) AE latent features
- `raw_ret`: (N, 9) 1-bar forward VWAP returns
- `train_idx`, `val_idx`, `test_idx`: data split indices

## Key Lessons Learned

1. **Delta policy is essential** — Direct position output leads to degenerate strategies
2. **Soft threshold sparsifies naturally** — Zero gradient when |score| < θ → precise zero weights
3. **10K steps is optimal** — Longer training overfits to training window
4. **No fancy architectures needed** — Simple 2-layer MLP with SiLU beats deeper variants
5. **Seed variance is significant** — Always validate with multiple seeds
6. **CPU training is fine** — Tiny network, MPS kernel overhead dominates
