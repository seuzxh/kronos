# 实验 3:Kronos 5min 特征提取器 + LGBM 选股集成

> **状态**:✅ 完成(首次成功)
> **日期**:2026-07-20 ~ 2026-07-21
> **核心结论**:Kronos 作为**反转特征提取器**注入 LGBM,OOS **RankIC 提升 6.4%**(0.0612 → 0.0651)。前两次端到端选股失败后,首次找到 Kronos 在 A 股选股上的有效用法。

## 一句话总结

不让 Kronos 端到端做选股(它不会),让它做**短期形态预测**,提取 8 个特征注入 LGBM。**微调版**特征使 LGBM 的横截面排序能力提升 6.4%,因为 Kronos 提供了与现有动量因子**互补的反转信号**。

## 三跑结果

| 跑 | 特征数 | Kronos 来源 | IC | RankIC | 判定 |
|---|:---:|---|:---:|:---:|:---:|
| A baseline | 18 | 无 | 0.0548 | 0.0612 | 对照 |
| B +base | 26 | Kronos-base | 0.0329 | 0.0439 | ❌ base 看跌偏差 |
| **C +finetune** | 26 | highbeta_5min 微调 | 0.0491 | **0.0651** | ✅ **RankIC +6.4%** |

## 关键洞察

1. **Kronos 特征本质是反转信号**:预测涨/振幅大 → 未来收益差(最强 `range` RankIC=-0.054)
2. **与现有动量因子互补**:LGBM 的 18 因子是动量,Kronos 提供反转维度
3. **微调是必要的**:base 版有 A 股看跌偏差(99% 预测收益 < +1.22%),反而拖累 LGBM;finetune 校准了偏差
4. **IC 下降但 RankIC 上升**:选股只关心排序,Kronos 改善排序但引入异常值扰动线性 IC

## 文档

- [设计详情](design.md)— 完整方案设计
- [提取报告](extraction-report.md)— 特征分布、base vs finetune 对比、单特征 RankIC
- [Ablation 报告](ablation-report.md)— 三跑对比、根因分析、结论

## 代码

- 提取脚本:`finetune/feature_extraction/`(含多卡并行 `extract_parallel.py`)
- LGBM 集成:`3.qlib_ifind_beta/qlib_ifind_beta/minute_enhanced_handler.py:MinuteEnhancedKronosHandler`
- Ablation yaml:`3.qlib_ifind_beta/qrun/workflow_kronos_ablation_{A,B,C}.yaml`

## GPU 加速

本实验用 **6 卡数据并行**(GPU 1-6),把特征提取从单卡 ~8 小时压到 **~100 分钟**(快 4.8 倍)。每张卡跑一个 worker,处理 1/6 的决策日,最后单进程合并写 bin。

## 后续优化

1. 特征筛选:只用 `range`、`uncertainty`、`low_ratio` 三个强信号
2. Winsorize 异常值,改善 IC
3. Kronos 反转 × 现有动量的交互项
4. 多周期(加日线 Kronos 特征)
