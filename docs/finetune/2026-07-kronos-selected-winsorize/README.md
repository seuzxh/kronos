# 实验 4:Kronos 特征精选 + Winsorize(判定性)

> **状态**:✅ 完成(判定性结论)
> **日期**:2026-07-21
> **核心结论**:精选 + Winsorize 比全量 Kronos 略好(RankIC 0.0651 → 0.0656),但提升微弱。**Kronos 特征的天花板约 RankIC +7%**,主要价值是修复 IC(去异常值)。后续应转向 **horizon 对齐**(中期反转)方向。

## 一句话总结

实验 3 加了 8 个 Kronos 特征后 RankIC +6.4%。诊断显示只有 3 个是真信号(其余是噪声),且异常值扰动 IC。本实验精选 3 个 + Winsorize 截尾,**测 Kronos 特征的真实天花板**。结果:RankIC 微弱提升(0.0651→0.0656),IC 明显恢复(0.0491→0.0553)。**Kronos 特征价值已接近天花板,精选+Winsorize 主要修复了异常值对线性相关的扰动**。

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
