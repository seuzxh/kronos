# Kronos 项目文档

> 本目录是 Kronos(金融 K 线基础模型)在本项目中的**学习资料 + 微调实验记录**。
> 所有文档按主题分类,便于新成员快速了解 Kronos 原理与历次微调方案的来龙去脉。

---

## 📚 目录导航

### 🧠 模型架构(学习了解 Kronos)

适合第一次接触本项目的人,从头读一遍即可理解 Kronos 是什么、怎么工作。

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [architecture/kronos-overview.md](architecture/kronos-overview.md) | Kronos 是什么、整体架构、训练目标、输入输出 | 所有人(先读这篇) |
| [architecture/tokenizer.md](architecture/tokenizer.md) | Tokenizer 把 K 线转成离散 token 的原理 | 想理解量化码本的人 |
| [architecture/predictor.md](architecture/predictor.md) | Predictor 自回归预测下一步 token | 想理解预测机制的人 |
| [architecture/finetune-pipeline.md](architecture/finetune-pipeline.md) | 本项目微调流水线(数据→训练→推理)设计 | 要跑微调实验的人 |

### 🧪 微调方案记录(每次实验的完整档案)

每次微调方案(不同股池/频率/预测目标)在此建档,记录**设计→训练→回测→结论**全链路。

| 方案 | 股池 | 频率 | 状态 | 结论 |
|---|---|---|---|---|
| [2026-07 高贝塔 5min](finetune/2026-07-highbeta-5min/README.md) | 高贝塔(883926.TI) | 5min | ❌ 已回测 | Val Loss 低(2.19)但 4 折回测稳定亏损(-32.55%), 无实战价值 |

➕ 新建方案请参考 [finetune/_TEMPLATE.md](finetune/_TEMPLATE.md) 模板。

---

## 🗂️ 其他相关文档

| 文档 | 位置 | 说明 |
|---|---|---|
| Kronos 原作者 README | [`/README.md`](../README.md) | 模型原始介绍、HuggingFace 权重、demo |
| CSV 微调教程 | [`/finetune_csv/README_CN.md`](../finetune_csv/README_CN.md) | 原仓库的单股票 CSV 微调流程(本项目在此基础上扩展) |
| Kronos 模型变体对比 | `/doc/*.md` | mini/small/base 三档配置对比、REST API、涨跌停规则 |

---

## 📝 文档约定

1. **微调方案命名**: `YYYY-MM-<简称>`,如 `2026-07-highbeta-5min`
2. **每个方案一个子目录**,内含 `README.md`(总览) + 若干分主题文档
3. **结论先行**: 每篇文档开头写「TL;DR / 核心结论」,再展开细节
4. **数据可复现**: 关键超参、数据路径、CV 切分、命令行都要写清楚
5. **诚实记录**: 失败的实验也要记,说明为什么不 work,避免重复踩坑
