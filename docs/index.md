# Kronos 量化实验笔记

> 用 **Kronos 金融基础模型**做 A 股高贝塔选股的三次完整实验记录 —— 原理、方法、失败、突破与所得。

[![Status](https://img.shields.io/badge/状态-3次实验找到正确用法-green)]()
[![Experiments](https://img.shields.io/badge/实验-2失败+1成功-important)]()
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/seuzxh/kronos/blob/master/LICENSE)

---

## 🎯 这里讲的是什么

这是一个**诚实的探索记录**。我们尝试用开源的 Kronos 模型(金融版的 GPT)做 A 股高贝塔选股:

- **实验 1、2(端到端选股)**:❌ 失败 —— Val Loss 降 42% 但 RankIC ≈ 0
- **实验 3(特征提取器)**:✅ **成功** —— Kronos 特征使 LGBM 的 RankIC +6.4%

关键转折:不是"Kronos 能不能选股",而是"**怎么用** Kronos"。端到端选股不行,但当**反转特征提取器**注入 LGBM,首次验证有效。

这份记录的价值:

- ✅ **讲清楚 Kronos 是什么、怎么工作**(架构原理)
- ✅ **建立正确的量化评估方法**(为什么"loss 低 ≠ 能赚钱")
- ✅ **明确 Kronos 端到端选股的能力边界**(避免重走弯路)
- ✅ **找到 Kronos 在 A 股选股的正确用法**(特征提取器,非端到端)
- ✅ **沉淀可复用的工程代码**(多卡并行特征提取、RankIC 评估、LGBM 集成)

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
- [实验 3:Kronos 特征+LGBM](finetune/2026-07-kronos-5min-features-lgbm/README.md) ✅ **特征提取器路线成功**

---

## ⚡ 一分钟版本

**做了什么**:探索用 Kronos 做 A 股高贝塔选股的三种方法。

**结果**:

| | 实验 1(5min) | 实验 2(日频) | **实验 3(特征提取器)** |
|---|---|---|---|
| 方法 | 端到端选股 | 端到端选股 | **Kronos 特征 → LGBM 排序** |
| 数据 | 4 个月 | 2.5 年 | 2.5 年 5min |
| 关键指标 | 4 折亏 -32.55% | RankIC -0.006 | **RankIC +6.4%** |
| 结论 | ❌ 失败 | ❌ 失败 | ✅ **成功** |

**实验 1、2 为什么失败**:Kronos 学会了"预测 K 线形态"(Val Loss 低),但**没学会"挑出能跑赢指数的股票"**(RankIC ≈ 0)。端到端选股是架构-任务错配。

**实验 3 为什么成功**:换思路 —— 不让 Kronos 端到端选股,让它当**反转特征提取器**,产出 8 个特征注入 LGBM。Kronos 提供了 LGBM 原有动量因子**缺失的反转维度**,两者互补。

**关键前提**:**必须微调**。预训练 base 版有 A 股看跌偏差,反而拖累 LGBM;微调版校准了偏差才有效。

---

## 🧭 那 Kronos 到底适合什么?

实验 3 证明:**端到端选股不行,但作为特征提取器有效**。详见 [**06 怎么用才有效**](tutorial/06-recommended-uses.md) 和 [**07 实验 3 突破**](tutorial/07-breakthrough-features.md)。

| 场景 | 适用度 | 说明 |
|---|---|---|
| 🟢 合成 K 线数据生成 | **强推荐** | 论文核心亮点(+22% 保真度),用于回测压力测试 |
| 🟢 波动率预测 | 推荐 | 期权/风控,BSQ 天然编码 OHLC 范围(MAE -9%)|
| 🟢 新市场冷启动 | 推荐 | zero-shot 跨市场能力是预训练核心价值 |
| 🟢 **A 股选股(特征提取器)** | **推荐(已验证)** | 实验 3:Kronos 特征 → LGBM,RankIC +6.4%。**必须微调** |
| 🟡 单标的价格预测 | 可尝试 | 趋势性强的标的(BTC、指数)|
| 🔴 **A 股选股(端到端)** | **不推荐** | 实验 1、2 证明(RankIC ≈ 0)|

---

## ⚠️ 风险提示

- 本文档**不构成投资建议**,所有回测均为历史数据
- 实验 3 的 RankIC +6.4% 是 **OOS 测试段(2026-04~07)** 结果,样本仅 3 个月,需更长 OOS 验证
- Kronos 特征的有效性依赖**微调**(base 版有 A 股偏差,会拖累 LGBM),生产用必须定期重训
- 结论基于 **A 股 + 高贝塔股池 + 5min** 的具体设置,其他市场/股池/频率可能有不同结论

---

## 📄 License

MIT License — 代码和文档均可自由使用,但需保留原作者署名。详见 [LICENSE](https://github.com/seuzxh/kronos/blob/master/LICENSE)。
