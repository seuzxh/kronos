# Kronos 量化实验笔记

> 用 **Kronos 金融基础模型**做 A 股高贝塔选股的两次完整实验记录 —— 原理、方法、失败与所得。

[![Status](https://img.shields.io/badge/状态-已归档归因-blue)]()
[![Experiments](https://img.shields.io/badge/实验-2次失败重要所得-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/seuzxh/kronos/blob/master/LICENSE)

---

## 🎯 这里讲的是什么

这是一个**诚实的失败记录**。我们尝试用开源的 Kronos 模型(金融版的 GPT)预测高贝塔股票走势做选股,做了两次实验,**两次都失败了**。

但这份记录的价值不在"做出赚钱策略",而在于:

- ✅ **讲清楚 Kronos 是什么、怎么工作**(架构原理)
- ✅ **建立正确的量化评估方法**(为什么"loss 低 ≠ 能赚钱")
- ✅ **明确 Kronos 在选股任务上的能力边界**(避免你重走弯路)
- ✅ **沉淀可复用的工程代码**(数据拉取、RankIC 评估、DDP 训练)

---

## 📚 怎么读这份文档

### 🚀 新手路径(推荐,约 30 分钟)

如果你不熟悉 Kronos 或量化,**从这里开始** —— 5 篇通俗报告,从零讲起:

1. [总览](tutorial/00-overview.md)
2. [01 什么是 Kronos](tutorial/01-what-is-kronos.md)— "K 线界的 GPT"
3. [02 我们想做什么](tutorial/02-what-we-tried.md)— 高贝塔、指增、微调
4. [03 怎么验证](tutorial/03-how-we-evaluated.md)— **最重要**:为什么 loss 低 ≠ 能赚钱
5. [04 实验结果](tutorial/04-results.md)— 实际数字
6. [05 我们学到什么](tutorial/05-lessons.md)— 结论与建议
7. [06 怎么用才有效](tutorial/06-recommended-uses.md)— **场景适配指南**:合成数据、波动率、选股的对比

### 🔬 深入 Kronos 架构

想理解 Kronos 的技术细节(tokenize、自回归、BSQ 量化器):

- [架构总览](architecture/kronos-overview.md)
- [Tokenizer 原理](architecture/tokenizer.md)
- [Predictor 原理](architecture/predictor.md)
- [微调流水线设计](architecture/finetune-pipeline.md)

### 📁 完整实验档案

每次实验的完整记录(设计/日志/训练报告/回测报告):

- [实验 1:高贝塔 5min](finetune/2026-07-highbeta-5min/README.md) ❌
- [实验 2:高贝塔指增日频](finetune/2026-07-highbeta-enhancement/README.md) ❌

---

## ⚡ 一分钟版本

**做了什么**:微调 Kronos-base,在 A 股高贝塔股池上预测股票走势,选预测涨幅最大的股票。

**结果**:两次实验都失败。

| | 实验 1(5min) | 实验 2(日频) |
|---|---|---|
| 数据 | 4 个月 | 2.5 年 |
| 微调效果 | Val Loss 降 6.8% | Val Loss 降 42% |
| 选股能力 | 4 折亏 -32.55% | 4 折平均 RankIC -0.006 |
| 结论 | ❌ 失败 | ❌ 失败 |

**为什么**:Kronos 学会了"预测 K 线形态"(Val Loss 低),但**没学会"挑出能跑赢指数的股票"**(RankIC ≈ 0)。这是两个不同的任务。

**建议**:别用 Kronos 做选股。走 LGBM + 手工因子路线(同一股池 IC 0.067 已验证有效)。

---

## 🧭 那 Kronos 到底适合什么?

选股失败 ≠ Kronos 没用。换对场景,它可能是当下最强的开源金融 TSFM。详见 [**06 怎么用才有效**](tutorial/06-recommended-uses.md)。

| 场景 | 适用度 | 说明 |
|---|---|---|
| 🟢 合成 K 线数据生成 | **强推荐** | 论文核心亮点(+22% 保真度),用于回测压力测试 |
| 🟢 波动率预测 | 推荐 | 期权/风控,BSQ 天然编码 OHLC 范围(MAE -9%)|
| 🟢 新市场冷启动 | 推荐 | zero-shot 跨市场能力是预训练核心价值 |
| 🟡 单标的价格预测 | 可尝试 | 趋势性强的标的(BTC、指数)|
| 🔴 **A 股横截面选股** | **不推荐** | 我们两次实验证明(RankIC ≈ 0)|

---

## ⚠️ 风险提示

- 本文档**不构成投资建议**,所有回测均为历史数据
- 结论基于 **A 股 + 高贝塔股池 + 日频/5min** 的具体设置,其他场景可能有不同结论
- Kronos 在其他任务(单标的预测、波动率、合成数据)上可能有效,本结论仅限于"横截面选股"

---

## 📄 License

MIT License — 代码和文档均可自由使用,但需保留原作者署名。详见 [LICENSE](https://github.com/seuzxh/kronos/blob/master/LICENSE)。
