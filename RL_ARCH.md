# SylvaQuant — RL Dynamic Allocation Architecture

## 核心思路

用 Flow Matching 学 (market_latent + PnL) 的联合演化，生成合成轨迹训练 RL 做动态仓位配置。

## 架构总览

```
  ┌──────────────────────────────────────────────────┐
  │               DATA PIPELINE (offline)            │
  │                                                  │
  │  62 features → RF → 9 signals → per-coin TH     │
  │                               ↓                  │
  │                     per-bar PnL (9 coins)        │
  │                               ↓                  │
  │                    Portfolio PnL (avg)           │
  └──────────────────────────────────────────────────┘
                          ↓
  ┌──────────────────────────────────────────────────┐
  │           STAGE 1: MARKET LATENT (AE)            │
  │                                                  │
  │  62 features × 9 coins → AE → 16-dim latent     │
  │  (per time step, unified market representation)  │
  │  Independent of #coins, independent of #features │
  └──────────────────────────────────────────────────┘
                          ↓
  ┌──────────────────────────────────────────────────┐
  │       STAGE 2: FLOW MATCHING (dynamics)          │
  │                                                  │
  │  Train flow on: (latent_t, PnL_t) → next state   │
  │  Loss: conditional flow matching (vector field)  │
  │  Output: flow model p(s_{t+1} | s_t)             │
  └──────────────────────────────────────────────────┘
                          ↓
  ┌──────────────────────────────────────────────────┐
  │        STAGE 3: SYNTHETIC TRAJECTORIES           │
  │                                                  │
  │  Sample initial state from real data             │
  │  Roll forward with flow model (Euler integration)│
  │  Generate 100K+ episodes × 200 steps             │
  │  ↓                                               │
  │  (latent, PnL) for each step — NO real data I/O  │
  └──────────────────────────────────────────────────┘
                          ↓
  ┌──────────────────────────────────────────────────┐
  │       STAGE 4: RL POLICY (dynamic weights)       │
  │                                                  │
  │  State:  [latent(16) + RF signals(9) + cur_w(9)] │
  │  Action: next_weights(9)  (softmax, ∈[0,1])      │
  │  Reward: portfolio_return(t)                     │
  │  Algo:   PPO (continuous action space)           │
  │  Train:  on synthetic trajectories only          │
  │  Time:   ~5 min (100K steps, no data loading)    │
  └──────────────────────────────────────────────────┘
```

## 组件说明

### Stage 1: Market Latent Encoder

**输入:** 62 features × 9 coins = 558 raw dims  
**输出:** 16-dim latent (unified market state)

```
558 → 256 → 128 → 64 → 16(z) → 64 → 128 → 256 → 558
             Dumbbell AE, SiLU, Dropout=0.1
             Loss = MSE(recon, input)
```

已有 `latent_mlp_ae.pq` 是1h 8-dim版本。
新版用15m数据 + 62 features × 9 coins，对齐RF信号。

每个时间步输出一个16维向量，表示整个市场状态。
与资产数、特征数无关 —— 截面任意多币都压缩到16维。

### Stage 2: Flow Matching

**输入:** s_t = (latent_16, PnL_1) = 17维状态向量  
**目标:** 学习 s_t → s_{t+1} 的连续时间向量场

```
Flow Model: MLP(17 → 128 → 128 → 17) with SiLU
Training: Conditional Flow Matching loss
          v_θ(s_t, t) = predicted direction
          L = ||v_θ(s_t, t) - (s_{t+1} - s_t)||²
```

训练数据：~39K 1h时间步（4.5年），每个样本是 (latent_t, PnL_t, latent_{t+1}, PnL_{t+1})

### Stage 3: 合成轨迹生成

从真实数据采初始状态，用Flow Model滚动生成新轨迹：

```
s_0 = sample from real data
for step in 1..200:
    v = flow_model(s_{step-1})
    s_step = s_{step-1} + v * dt
```

生成100条×200步 = 20K样本，可重复采样。
**关键:** 生成过程纯CPU/GPU运算，无磁盘I/O，数秒钟完成。

### Stage 4: RL Agent

**状态空间:** (26维)
- market_latent (16) — 全局市场状态
- RF signals (9) — 各币当前信号强度
- current_weights (9) — 当前仓位占比

**动作空间:** (9维)
- 下一期各币权重 w_i ∈ [0, 1], Σw_i = 1

**奖励:**
- 每步: r_t = Σ(w_i * coin_return_i) — 加权组合收益
- 可加项: r_t - λ * |w_t - w_{t-1}| (换仓惩罚)

**训练:**
- PPO, batch=256, lr=3e-4
- 100K steps on synthetic data
- ~5分钟跑完

## 与当前系统对比

| 维度 | 当前(等权) | RL动态配置 |
|:----|:----------:|:----------:|
| 权重 | 1/9 each | 学出来的，可大可小 |
| 对BTC弱势期 | 硬扛 | 自动降低权重 |
| 对altcoin强势 | 均分 | 自动放大 |
| 换仓 | 无 | 有换仓成本，RL自己权衡 |
| 训练时间 | 0 | ~5分钟(flow) + ~5分钟(RL) |

## 实施步骤

```
Week 1: AE market latent (复用现有架构, 对齐15m)
Week 2: Flow matching training & validation
Week 3: RL policy on synthetic data
Week 4: Integration → CPCV evaluation → compare vs equal weight
```
