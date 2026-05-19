# VWAP Timing Convention (signal_return_offset=2)

这个文档记录 RL 实验中 VWAP return 的正确时序偏移，避免反复踩坑。

## 核心规则

```
Bar t:  特征 → latent[t] → policy → weights[t]
Bar t+1: 以VWAP[t+1]开仓 (entry_offset=1)
Bar t+2: 以VWAP[t+2]平仓，收益 = VWAP[t+2]/VWAP[t+1]-1
```

**Reward[t] = Σ weights[t]_i × (VWAP[t+2]_i / VWAP[t+1]_i - 1)**

## 代码实现

```python
# 正确 (offset=2):
Nret = len(latents) - 2
for j in range(NC):
    r = vwap[2:, j] / vwap[1:-1, j] - 1.0
    ret[:, j] = np.where(np.isfinite(r), r, 0.0)

# 错误 (offset=1, 有同bar泄漏):
ret[t] = vwap[t] / vwap[t-1] - 1  # ✗ 暴露了VWAP[t]给latent[t]

# 错误 (offset=0, 用close):
ret[t] = close[t] / close[t-1] - 1  # ✗ 不是VWAP fill时序
```

## 为什么是 offset=2？

来自 `pipeline_cpcv.py` 的 `backtest_pnl()` 函数：

```python
# VWAP fill timing:
# signal[i] → position[i+2] (在VWAP[i+1]开仓, VWAP[i+2]平仓)
# return[i+2] = VWAP[i+2] / VWAP[i+1] - 1

P[2:] = np.where(signals[:-2] > threshold, 1.0, ...)  # 位置在t+2
R[1:] = vwaps[1:] / vwaps[:-1] - 1                     # 收益 = vwap[t]/vwap[t-1]-1
pnl = P * R
```

- `P[t]` = signal[t-2] 决定的仓位
- `R[t]` = vwap[t]/vwap[t-1]-1 的收益
- 所以 `pnl[t]` = 用 signal[t-2] 开仓，在 VWAP[t-1]→VWAP[t] 的收益

等价于：
- signal 在 bar t → position 在 bar t+2
- return = vwap[t+2]/vwap[t+1]-1

## 数据对齐

当数据形状变化时，latent和return的对齐方式：

```python
latents.shape = (N, 16)
returns.shape = (N-2, 9)  # 少了2个bar，因为需要VWAP[t+2]来计算return[t]

# 正确对齐：用 latents[0:N-2] 对应 returns[0:N-2]
lat_align = latents[:-2]   # (N-2, 16) — 丢弃最后2个latent
# lat_align[t] = latents[t] (bar t的特征)
# returns[t] = VWAP[t+2]/VWAP[t+1]-1 (2个bar后的收益)
# → latent[t] 不包含 VWAP[t+2] 的信息 → 无泄漏
```

## 验证方法

快速检查是否存在同bar泄漏：

```python
# 检查latent[t]和return[t]的相关性
# 如果offset正确，相关性应接近0
import numpy as np
latents = np.load('data/market_latent.npy')
# 正确的reward: offset=2
ret_correct = vwap[2:] / vwap[1:-1] - 1  # return[t]用VWAP[t+2]
lat_correct = latents[:-2]                # latent[t]用bar t
# 快速验证：latents[:,-1] × ret_correct的第1列的相关性
corr = np.corrcoef(lat_correct[:1000, 0], ret_correct[:1000, 0])[0,1]
print(f'Correct offset corr: {corr:.3f} (should be ~0)')

## 相关代码位置

- `pipeline_cpcv.py:296` — `entry_offset=1, signal_return_offset=2`
- `pipeline_cpcv.py:325` — `eval_ret[:-2] = price[2:] / price[1:-1] - 1`
- `pipeline_cpcv.py:620-638` — `backtest_pnl()` VWAP fill timing
- `train_rl_policy.py:38-41` — 正确 reward 计算
- `experiments/rl-variants/00_prepare_data.py` — 实验数据准备

## Pitfall

**症状**: 用 `VWAP[t]/VWAP[t-1]-1` 做reward，Policy得到超高Sharpe（>50）
**根因**: latent[t] 用了 volume/quote_vol，可计算 VWAP[t]，reward 也用了 VWAP[t] → **同bar泄漏**
**修复**: reward 用 `VWAP[t+2]/VWAP[t+1]-1`
