# 实验 4:Kronos 特征精选 + Winsorize(判定性 + 纠错实验 3)

> **状态**:✅ 完成 —— **推翻了实验 3 的"成功"结论**
> **日期**:2026-07-21
> **核心结论**:精选 + Winsorize 让 RankIC +7.2%(看似更好),但**完整回测显示年化超额暴跌 65%**(143%→78%),夏普腰斩(3.25→1.83)。**Kronos 特征对 A 股高贝塔选股是净负面的**。

## 一句话总结

实验 3 曾基于 RankIC +6.4% 宣称"特征提取器路线成功"。本实验补做完整回测后发现:**Kronos 特征(不论全量/精选/Winsorize)都拖累了超额收益**。RankIC 提升是统计假象,无法转化为实际 alpha。这是一次真实的自我纠错。

## 四跑对比(含完整回测)

| 跑 | 特征数 | 处理 | IC | RankIC | 年化超额 | 夏普 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A baseline** | 18 | 无 | 0.0548 | 0.0612 | **+143.67%** | **3.25** |
| B +base | 26 | 无 | 0.0329 | 0.0439 | +45.21% | 0.96 |
| C +全 finetune | 26 | 无 | 0.0491 | 0.0651 | +127.53% | 2.93 |
| **D +精选+Winsorize** | 21 | 3 字段截尾 | 0.0553 | 0.0656 | +78.65% | 1.83 |

**RankIC 最高的 D 跑,超额收益反而最差**(从 143% 暴跌到 78%)。这印证了"RankIC 提升 ≠ 能赚钱"。

## 四跑对比

| 跑 | 特征数 | 处理 | IC | RankIC | vs A |
|---|:---:|:---:|:---:|:---:|:---:|
| **A baseline** | 18 | 无 | 0.0548 | 0.0612 | — |
| B +Kronos-base | 26 | 无 | 0.0329 | 0.0439 | ❌ base 看跌偏差 |
| C +全 Kronos-finetune | 26 | 无 | 0.0491 | 0.0651 | RankIC +6.4% |
| **D +精选 Kronos+Winsorize** | 21 | 3 字段截尾 | **0.0553** | **0.0656** | RankIC **+7.2%** |

## 判定结论

| 维度 | 结论 |
|---|---|
| D vs C 的 RankIC 提升 | +0.0005(微弱)→ 精选 3 个 vs 全量 8 个,横截面排序几乎无差 |
| D vs C 的 IC 提升 | +0.0062(明显)→ Winsorize 确实去除了异常值对线性 IC 的扰动 |
| D 的 IC 恢复到 baseline | 0.0553 ≈ 0.0548 → 异常值问题已解决 |
| Kronos 真信号天花板 | RankIC +7% 左右,**精选+Winsorize 不会改变这个边界** |

**核心洞察**:
- LGBM 对冗余/噪声特征有鲁棒性 —— 8 个 vs 3 个,RankIC 几乎一样
- Winsorize 修复了 IC(线性相关),但不改善 RankIC(排序)
- **Kronos 特征的天花板就在 RankIC +7% 附近**,想突破得换思路(中期反转方向)

## 文档

- [设计详情](design.md)— 精选逻辑 + Winsorize 设计
- [Ablation 报告](ablation-report.md)— 四跑对比 + 根因分析 + 后续方向

## 代码

- Winsorize processor:`3.qlib_ifind_beta/qlib_ifind_beta/processors.py`
- 精选 handler:`3.qlib_ifind_beta/qlib_ifind_beta/minute_enhanced_handler.py:MinuteEnhancedKronosSelectedHandler`
- D 跑 yaml:`3.qlib_ifind_beta/qrun/workflow_kronos_ablation_D_selected.yaml`

## 后续方向(重要)

本实验是个**判定性节点**:Kronos 特征在当前框架(T+1 label、5min 周期、LGBM 排序)下的天花板已明确,约 RankIC +7%。

要进一步突破,必须换框架,最可能有效的方向是:

1. **中期反转策略(最推荐)**:Kronos 的 `range` 特征在 T+10 的 RankIC 是 T+1 的 3 倍(-0.054 → -0.160)。把 label 从 T+1 改成 T+5/T+10,理论上 RankIC 可能 +20% 以上
2. **多尺度融合**:加日线 Kronos 特征(已有 checkpoint),捕捉不同周期的反转
3. **特征交互**:Kronos 反转 × 现有动量的交互项,捕捉"动量到反转的拐点"
