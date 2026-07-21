# 实验 3:Kronos 5min 特征提取器 + LGBM 选股集成

> **状态**:⚠️ **已纠错**(原结论"成功"被推翻)
> **日期**:2026-07-20 ~ 2026-07-21
> **原结论**(错误):基于 RankIC +6.4% 宣称"首次成功"
> **纠正结论**(2026-07-21):补充完整回测后发现,**Kronos 特征拖累了超额收益**(年化 -16%),RankIC 提升是统计假象,无法转化为实际 alpha。

## 一句话总结(纠正版)

不让 Kronos 端到端做选股,让它做特征提取器注入 LGBM。**RankIC 提升了 6.4%,但完整回测显示超额收益下降了 16%**。这印证了"RankIC 提升 ≠ 能赚钱"——我自己犯了我曾经警告过的错误(详见 [实验 4 纠错](../2026-07-kronos-selected-winsorize/ablation-report.md))。

## 三跑结果(含完整回测)

| 跑 | 特征数 | Kronos 来源 | IC | RankIC | 年化超额 | 夏普 |
|---|:---:|---|:---:|:---:|:---:|:---:|
| A baseline | 18 | 无 | 0.0548 | 0.0612 | **+143.67%** | **3.25** |
| B +base | 26 | Kronos-base | 0.0329 | 0.0439 | +45.21% | 0.96 |
| C +finetune | 26 | highbeta_5min 微调 | 0.0491 | 0.0651 | +127.53% | 2.93 |

**原判定**(错误,只看 RankIC):C 跑 RankIC +6.4% 达标 → "成功"
**纠正判定**(完整回测):C 跑超额收益 -16%、夏普 -0.32 → **实际拖累了策略**

## 关键洞察(部分仍成立)

1. **Kronos 特征本质是反转信号**:预测涨/振幅大 → 未来收益差(最强 `range` RankIC=-0.054)—— 这个事实仍成立
2. **与现有动量因子统计互补**:但统计互补 ≠ 实际超额提升
3. **微调校准了 base 偏差**:base 版看跌偏差(99% 预测收益 < +1.22%),finetune 校准了
4. **⚠️ 但 RankIC 提升不等于赚钱**:这正是 [03 怎么验证](../../tutorial/03-how-we-evaluated.md) 的核心教训,我自己犯了的错误

## 为什么原结论错了

原 ablation 只看 IC/RankIC,因为 qlib 的 `PortAnaRecord` 在本环境有 bug(`inst_processors` 冲突),回测阶段失败。我为省时间 kill 了回测,**只留 IC/RankIC 就宣称成功**。

补写 `backtest_ablation.py`(用 pred.pkl + label.pkl 自己算)后,真相暴露:RankIC 的提升在 top-10 选股层面不成立,反而拖累超额收益。

详细分析见 [实验 4 纠错报告](../2026-07-kronos-selected-winsorize/ablation-report.md)。

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
