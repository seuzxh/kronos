# Kronos 量化实验笔记

> 用 **Kronos 金融基础模型**做 A 股高贝塔选股的四次完整实验记录 —— 原理、方法、失败、纠错与所得。

[![Status](https://img.shields.io/badge/状态-4次实验均无实际alpha-red)]()
[![Experiments](https://img.shields.io/badge/实验-4次(含1次自我纠错)-important)]()
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/seuzxh/kronos/blob/master/LICENSE)

---

## 🎯 这里讲的是什么

这是一个**诚实的失败 + 自我纠错记录**。我们尝试用开源的 Kronos 模型做 A 股高贝塔选股,做了四次实验:

- **实验 1、2(端到端选股)**:❌ 失败 —— Val Loss 降 42% 但 RankIC ≈ 0
- **实验 3(特征提取器)**:⚠️ 原宣称"RankIC +6.4% 成功",**后被实验 4 纠错**
- **实验 4(精选+Winsorize+完整回测)**:❌ Kronos 特征**拖累超额收益**(年化 -16% ~ -65%)

**最终结论**:**Kronos 对 A 股高贝塔选股没有实际 alpha**(四次实验一致)。但这个结论本身有价值 —— 它是"RankIC 提升 ≠ 能赚钱"的活教材。

这份记录的价值:

- ✅ **讲清楚 Kronos 是什么、怎么工作**(架构原理)
- ✅ **建立正确的量化评估方法**(为什么"loss 低 ≠ 能赚钱"、**"RankIC 提升 ≠ 能赚钱"**)
- ✅ **明确 Kronos 对 A 股选股的能力边界**(四次实验,盖棺定论)
- ✅ **一次真实的自我纠错**(实验 4 推翻了实验 3 的错误结论)
- ✅ **沉淀可复用的工程代码**(多卡并行特征提取、回测框架、LGBM 集成)

---

## 📚 怎么读这份文档

### 🚀 新手路径(推荐,约 30 分钟)

如果你不熟悉 Kronos 或量化,**从这里开始** —— 通俗报告,从零讲起:

1. [总览](tutorial/00-overview.md)
2. [01 什么是 Kronos](tutorial/01-what-is-kronos.md)— "K 线界的 GPT"
3. [02 我们想做什么](tutorial/02-what-we-tried.md)— 高贝塔、指增、微调
4. [03 怎么验证](tutorial/03-how-we-evaluated.md)— **最重要**:为什么 loss 低 ≠ 能赚钱
5. [04 实验结果](tutorial/04-results.md)— 实际数字(实验 1、2)
6. [05 我们学到什么](tutorial/05-lessons.md)— 结论与建议
7. [06 怎么用才有效](tutorial/06-recommended-uses.md)— **场景适配指南**
8. [07 实验 3 突破:特征提取器路线](tutorial/07-breakthrough-features.md)— **从失败到成功的转折**

### 🔬 深入 Kronos 架构

想理解 Kronos 的技术细节(tokenize、自回归、BSQ 量化器):

- [架构总览](architecture/kronos-overview.md)
- [Tokenizer 原理](architecture/tokenizer.md)
- [Predictor 原理](architecture/predictor.md)
- [微调流水线设计](architecture/finetune-pipeline.md)

### 📁 完整实验档案

每次实验的完整记录(设计/日志/训练报告/回测报告):

- [实验 1:高贝塔 5min](finetune/2026-07-highbeta-5min/README.md) ❌ 端到端选股失败
- [实验 2:高贝塔指增日频](finetune/2026-07-highbeta-enhancement/README.md) ❌ 端到端选股失败
- [实验 3:Kronos 特征+LGBM](finetune/2026-07-kronos-5min-features-lgbm/README.md) ⚠️ 原报成功,后纠错
- [实验 4:Kronos 精选+Winsorize](finetune/2026-07-kronos-selected-winsorize/README.md) ❌ 完整回测后失败 + 纠错

---

## ⚡ 一分钟版本

**做了什么**:探索用 Kronos 做 A 股高贝塔选股的多种方法。

**结果**(含完整回测,2026-07-21 纠错后):

| | 实验 1(5min) | 实验 2(日频) | 实验 3(特征提取器) | 实验 4(精选+Wins) |
|---|---|---|---|---|
| 方法 | 端到端选股 | 端到端选股 | Kronos 特征→LGBM | 精选+Winsorize |
| 关键指标 | 4 折亏 -32.55% | RankIC -0.006 | RankIC +6.4% 但**超额 -16%** | RankIC +7.2% 但**超额 -65%** |
| 结论 | ❌ 失败 | ❌ 失败 | ❌ 失败(原误报成功) | ❌ 失败(纠错后) |

**最终结论**:**Kronos 对 A 股高贝塔选股没有实际 alpha**。四次实验一致指向同一结论。

**实验 3 的教训(最重要)**:RankIC 提升(+6.4%)不等于超额收益提升(-16%)。我自己犯了我曾经在 [03 怎么验证](tutorial/03-how-we-evaluated.md) 里警告过的错误 —— 只看 IC/RankIC 就宣称成功,没跑完整回测。实验 4 纠正了这个错误。

**Kronos 不是没用**:它在其他场景(合成数据、波动率、单标的预测)可能有效,详见 [06 怎么用才有效](tutorial/06-recommended-uses.md)。但对 A 股横截面选股,四次实验盖棺定论。

---

## 🧭 那 Kronos 到底适合什么?

四次实验一致证明:**Kronos 对 A 股选股没有实际 alpha**(不论端到端还是特征提取器)。但 Kronos 在其他场景可能有效。详见 [**06 怎么用才有效**](tutorial/06-recommended-uses.md)。

| 场景 | 适用度 | 说明 |
|---|---|---|
| 🟢 合成 K 线数据生成 | **强推荐** | 论文核心亮点(+22% 保真度),用于回测压力测试 |
| 🟢 波动率预测 | 推荐 | 期权/风控,BSQ 天然编码 OHLC 范围(MAE -9%)|
| 🟢 新市场冷启动 | 推荐 | zero-shot 跨市场能力是预训练核心价值 |
| 🟡 单标的价格预测 | 可尝试 | 趋势性强的标的(BTC、指数)|
| 🔴 **A 股横截面选股** | **不推荐** | **四次实验均失败**(实验 1/2 RankIC≈0,实验 3/4 超额下降)|

---

## ⚠️ 风险提示

- 本文档**不构成投资建议**,所有回测均为历史数据
- 实验 3 原宣称"成功"的结论**已被实验 4 纠错**(见 [实验 4 纠错报告](finetune/2026-07-kronos-selected-winsorize/ablation-report.md))
- 四次实验的 OOS test 段都只有 2026-04~07(62 天),样本短,结论需谨慎
- "Kronos 对 A 股选股无 alpha"的结论限于 **A 股 + 高贝塔股池 + 5min/日频** 设置

---

## 📄 License

MIT License — 代码和文档均可自由使用,但需保留原作者署名。详见 [LICENSE](https://github.com/seuzxh/kronos/blob/master/LICENSE)。
