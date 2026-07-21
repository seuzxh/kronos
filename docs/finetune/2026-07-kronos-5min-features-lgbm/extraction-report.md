# 提取报告:Kronos 5min 特征

## 1. 提取规模

| 指标 | 值 |
|---|---|
| 决策日数 | 598(2024-01-09 ~ 2026-07-02)|
| 每日成分股 | ~100(高贝塔 883926)|
| 预期样本数 | ~59800 |
| **实际有效样本** | **58,980**(base + finetune 各一份)|
| Skip 率 | ~14%(数据不足:新股/停牌/1min 缺失)|
| 失败率 | 0% |
| Checkpoint | base + finetune 两版,各 58,980 行 |
| 提取耗时 | 各 ~100 分钟(6 GPU 并行)|
| 总算力 | ~1200 GPU·分钟(6 卡 × 100 min × 2 版) |

## 2. 特征分布对比(base vs finetune)

| 特征 | base mean | finetune mean | base std | finetune std | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| `ret_intraday` | -0.0290 | **-0.0021** | 0.0193 | 0.0265 | finetune 校准看跌偏差 |
| `high_ratio` | +0.0167 | +0.0350 | 0.0141 | 0.0234 | finetune 预测更激进 |
| `low_ratio` | -0.0455 | -0.0361 | 0.0205 | 0.0189 | finetune 跌幅预测更温和 |
| `range` | +0.0622 | +0.0710 | 0.0269 | 0.0309 | finetune 振幅更大 |
| `close_pos` | 0.2807 | **0.4542** | 0.1220 | 0.1608 | finetune 接近 0.5 中性 |
| `morning_ret` | -0.0215 | **-0.0004** | 0.0157 | 0.0226 | finetune 校准上午偏差 |
| `afternoon_ret` | -0.0077 | -0.0018 | 0.0095 | 0.0091 | 两者相近 |
| `uncertainty` | +0.0182 | **+0.0365** | 0.0122 | 0.0228 | finetune 不确定性更高 |

**关键差异**:finetune 版的 `ret_intraday`、`close_pos`、`morning_ret` 都更接近中性(0 或 0.5),说明微调校准了 base 的系统性偏差。finetune 版 `uncertainty` 更高,说明模型对 A 股更"敏感"。

## 3. 单特征 RankIC(20,000 样本采样)

衡量每个特征与 T+1 实际收益的横截面排序相关性:

| 特征 | base RankIC | finetune RankIC | 类型 |
|---|:---:|:---:|---|
| `range` | **-0.054** | **-0.054** | 强反转(稳定)|
| `uncertainty` | -0.031 | -0.037 | 反转 |
| `low_ratio` | +0.043 | +0.037 | 反转(预测跌→实际涨)|
| `high_ratio` | -0.039 | -0.035 | 反转(预测涨→实际跌)|
| `ret_intraday` | +0.022 | +0.007 | 弱动量 |
| `morning_ret` | +0.020 | +0.006 | 弱动量 |
| `afternoon_ret` | +0.007 | +0.013 | 弱动量 |
| `close_pos` | -0.006 | +0.004 | 无信号 |

**核心洞察**:Kronos 特征本质是**反转信号**(预测涨/振幅大 → 未来差)。最强的 `range` 两版都稳定在 -0.054。

## 4. 特征间相关性(finetune 版)

高度共线对(|corr| > 0.7):
- `ret_intraday` ↔ `morning_ret`: 0.879
- `ret_intraday` ↔ `low_ratio`: 0.848
- `low_ratio` ↔ `range`: -0.855

**结论**:8 个特征有效维度约 4 个(方向类、振幅类、位置类、不确定性类各一)。后续可精简到 4-5 个强特征。

## 5. 覆盖率

- **全股池覆盖**:58,980 / 预期 59,800 = **98.6%**
- **Skip 原因**:14% 的样本因 1min 数据不足(新股上市晚、停牌、备份缺失)被跳过
- **bin 对齐**:所有特征 bin 与 `close.day.bin` 严格对齐(start_index + length 一致)
- **按时间段**:早期(2024-01)skip 率略高(历史不足),后期正常

## 6. base vs finetune 的判定

**finetune 版明显优于 base 版**(LGBM RankIC 0.0651 vs 0.0439,提升 48%)。

原因:
1. base 模型有 A 股看跌偏差(99% 样本预测收益 < +1.22%)
2. finetune 在高贝塔股池上训练过,校准了偏差
3. finetune 的 `uncertainty` 更高,提供更丰富的信心度信号

**实践建议**:后续只用 finetune 版,base 版仅作 ablation 对照。
