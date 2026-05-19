# GRPO RL Training — Experiment Log

> 文件规划：每次成功迭代存为独立文件，禁止覆盖已有记录。
> 实验目录: `experiments/rl-variants/`, 稳定版: `src/train_rl_policy.py`

---

## ✅ v5 — GRPO Gate (🏆 最终冠军)
**文件**: `experiments/rl-variants/exp_r_grpo_gate.py`  
**Commit**: eff8032 (2026-05-18)

### 核心设计
- Policy 双头输出: **direction** (tanh, [-1,1]) × **gate** (sigmoid, [0,1])
- Gate 初始化 bias=-5 → sigmoid≈0.007 → 默认关闭
- 入场费: `FEE × |w| = FEE × gate × |direction|` → gate≈0 时不花钱
- 合成 CFM 数据训练, 实盘 market latent 验证

### 结果 (实盘 OOS)
| 指标 | 值 |
|------|-----|
| Gross SR | 2.38 |
| **Net SR** | **+1.19** ✅ |
| Σ\|w\| | 0.11 (对比原版 8.71, 缩小 79×) |
| TO | 0.06 (对比原版 2.61, 降低 43×) |
| gate | 0.011±0.009 (≈1% 开仓) |
| σ | 0.61 (收敛) |
| 训练时间 | 163s / 5000步 |

### 成功原因
- Gate + 入场费 = **自然稀疏性**。只有确信预期收益 > 0.04% 入场费时才开仓
- 合成数据中有极端收益事件 → gate 学会只在那些时刻打开
- 低 TO (0.06) 让 fee 成本可忽略
- 首次实现费后正夏普

### 教训
- 之前反复删改同一文件，导致版本丢失
- 每次实验应存为独立文件

---

## v4 — Path-Level GRPO (失败)
**文件**: `experiments/rl-variants/exp_r_grpo_chunk.py` (被覆盖过)

### 设计
- L=10 步路径累积回报, B=2048 并行程的组间归一化
- PathEnv 类, 每步 w_prev 跟踪

### 结果
| 指标 | 值 |
|------|-----|
| Gross SR | -1.28 ❌ |
| Net SR | -4.37 |
| TO | 1.63 |
| σ | 0.42↓ |

### 失败原因
- 路径级信号信噪比更低：L 步累积噪声增长 √L 而非 L
- 组间归一化在噪声环境下学出错误策略

---

## v3 — 实盘 AE 数据 + 各种 Reward (探索)

### 实验系列
多个 reward 设计在实盘 AE latent 上对比:
- Pure w·ret → Gross SR=2.39, Net SR=-3.53, TO=4.64
- w·ret - fee×\|Δw\| → Gross SR=1.13, TO=1.17
- r - λ₁·max(0,-r) - λ₂·r² → Gross SR=2.43, Net SR=-4.89, TO=2.65

### 结论
- 实盘 15m crypto 每 bar edge ≈ 0.01%, σ ≈ 0.46%
- Fee 0.04% 是收益的 4-18×, 任何连续权重策略都被 fee 拖垮
- **15m 数据 + 连续权重 = 不适用于 fee 敏感的 RL**

---

## v2 — 去 Fee (之前的最佳 RF 版本)
**文件**: `src/train_rl_policy.py` (commit b5c47eb) — 原始版去除 fee

### 修改
- 删除 FEE 常量和所有 fee 相关代码
- Reward = pure w·ret (no fee, no entry cost)

### 坑
- 合成数据是 i.i.d. 的，段间无真实时序连续性
- "fee 在 i.i.d. 合成数据上 = L1 惩罚, 非真实换手率约束"
- 去 fee 后合成数据 SR↑ (3.37→4.07), 但实盘泛化不变

---

## v1 — 原始 GRPO (带 Fee)
**文件**: `src/train_rl_policy.py` (commit c3d1386)

### 设计
- 合成 CFM 数据 (`data/synthetic_cfm.npz`), 非重叠段 L=5
- B=4000 段并行, 每步 K=32 采样, 组内归一化
- Fee 结构: l==0 入场费 FEE×\|w\|, l>0 换手费 FEE×\|Δw\|
- w_prev 段内跟踪
- 验证在实盘 market latent 上

### 结果
- Gross SR=3.09 (本次) / 4.07 (上次)
- 但 TO=2.61, fee 成本 = 0.001044/bar >> edge=0.000642/bar
- Net SR 负

### 核心教训
- 合成数据训练 + 实盘验证的模式有效 (信号迁移)
- 但 fee 结构不足以强制稀疏交易
- w_prev 段内连续, 但段间不连续

---

## 关键技术原则

1. **实验分离**: 每次实验存为 `experiments/rl-variants/exp_<描述>.py`, 禁止覆盖
2. **稳定版在 `src/`**: `src/train_rl_policy.py` 只保留 git 控制的生产版本
3. **Gate 机制**: direction × gate = 自然稀疏, 入场费自动约束
4. **Synthetic → Real**: 合成数据训练, 实盘验证, 信号迁移能力强
5. **Net SR > Gross SR**: 低 TO 比高 Gross SR 更重要 (fee 吃掉一切)

---

## 🏆 最终结论

**冠军方案: GRPO Gate**
- 文件: `exp_r_grpo_gate.py`
- Net SR=+1.19, TO=0.06, Σ|w|=0.11

**核心机制**: direction(tanh) × gate(sigmoid, init bias=-5) + 入场费 FEE×|w|
→ gate≈0 时无仓位无成本 → 只在确信时开仓 → 自然稀疏

**成功因素**:
1. 合成 CFM 数据训练（强信号, 极端事件丰富）
2. Gate 初始关闭（bias=-5），需学习打开
3. 入场费结构：开仓先交 0.04%，迫使策略只在高确信时出手
4. 实盘 infer 验证信号迁移

**实验清理**: 所有中间版本移入 `archive/`，仅保留冠军版。
