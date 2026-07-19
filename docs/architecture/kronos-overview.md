# Kronos 模型架构总览

> 适合第一次接触本项目的人。读完这篇你会知道:Kronos 是什么、解决什么问题、整体怎么工作。
> 后续深入阅读:[tokenizer.md](tokenizer.md) / [predictor.md](predictor.md) / [finetune-pipeline.md](finetune-pipeline.md)

---

## TL;DR

- **Kronos 是首个开源的金融 K 线(K-line / candlestick)基础模型**,把"预测股价走势"建模成**类语言模型预测下一个 token**的任务。
- 训练数据来自全球 45+ 交易所,逐根 K 线被**量化成离散 token**,然后用 Transformer 自回归预测未来 token,再解码回 K 线。
- 两大核心组件:
  1. **Tokenizer** —— 把连续的 OHLCV K 线 → 离散 token(可逆,能解码回 K 线)
  2. **Predictor** —— GPT 风格的 Transformer,给定历史 token 序列预测未来 token
- 本项目(Kronos-base)在 A 股高贝塔股池上做了 **5min 频率微调**,验证其能力边界。

---

## 1. 为什么用"语言模型"思路预测 K 线?

传统量化把股价当**连续数字**回归。Kronos 借鉴 LLM 思路,把它当**离散符号序列**建模:

| 维度 | 连续回归(传统) | 离散 token(Kronos) |
|---|---|---|
| 输入 | `[95.76, 96.20, ...]` 浮点 | `[token_42, token_7, ...]` 整数 |
| 预测 | 均值 + 方差(常假设高斯) | 概率分布 over 码本(任意形状) |
| 优势 | 简单 | 能拟合厚尾、多模态分布(金融典型特征)|
| 输出 | 单点 + 置信区间 | 整条 K 线形态(可采样多条) |

**关键洞察**:金融收益分布是**非高斯**的(尖峰厚尾、偶尔跳变),用离散 token + softmax 概率能更自然地表达这种分布。

---

## 2. 整体架构(数据流)

```
                  ┌─────────────────────────────────────────────────┐
   历史 K 线      │  1. Tokenizer.encode()                          │
  (OHLCV + 时间) ─┼─→  2. 量化成 token 序列 [s1_id, s2_id, ...]      │
                  │  3. Predictor.forward() / generate()            │
                  │     自回归预测未来 token                         │
                  │  4. Tokenizer.decode()                          │
   未来 K 线     ←┼─→  5. token → 连续值 → 反归一化 → 预测 K 线      │
  (OHLCV)        │                                                  │
                  └─────────────────────────────────────────────────┘
```

**输入**:一个长度为 `lookback` 的窗口,每根 K 线含 6 维特征 `[open, high, low, close, volume, amount]` + 5 维时间特征 `[minute, hour, weekday, day, month]`。

**输出**:长度为 `pred_len` 的未来 K 线预测,采样 `sample_count` 次取均值得概率预测。

---

## 3. 两大组件详解

### 3.1 Tokenizer(`KronosTokenizer`)

作用:**K 线 ↔ token** 的可逆映射。详见 [tokenizer.md](tokenizer.md)。

- 架构:Encoder-Decoder Transformer + BSQ(Binary Spherical Quantizer)量化器
- 把每根 K 线压成 **2 个 token**:`s1`(主语义,12 bit)+ `s2`(残差细节,4 bit)
- 训练目标:重建误差(MSE) + 码本利用率熵正则

### 3.2 Predictor(`Kronos`)

作用:在 token 空间做自回归预测。详见 [predictor.md](predictor.md)。

- 架构:GPT 风格 decoder-only Transformer(RoPE 位置编码)
- 输入历史 token,预测下一个 token 的概率分布
- 损失:s1/s2 双头交叉熵
- 推理:温度采样 + top-p,可并行采样多条

---

## 4. 三档模型变体

HuggingFace 上有三档预训练权重(本项目用 base):

| 变体 | 参数量 | max_context | Tokenizer | 适合 |
|---|---|---|---|---|
| **mini** | ~25M | 2048 | Tokenizer-2k | 长上下文、快速实验 |
| **small** | ~45M | 512 | Tokenizer-base | 中等任务 |
| **base** | **102M** | 512 | Tokenizer-base | **本项目使用**,精度最高 |

> ⚠️ max_context 限制 `lookback + predict + 1 ≤ 512`,本项目用 240+48+1=289,在 base 上没问题。

---

## 5. 本项目用它做了什么

本项目把 Kronos-base 当**特征提取器/预测器**,在 A 股上做微调实验:

1. **首次微调方案**(2026-07):高贝塔股池(883926.TI)5min 频率,预测次日全天 48 根 5min K 线
2. **流程**:1min K 线聚合 → 5min → 4 折 CV 微调 → 日级调仓回测
3. **结论**:Val Loss 下降明显(2.35→2.19),但回测稳定亏损,**证明"形态学得会 ≠ 选股 alpha"**

详见 [finetune/2026-07-highbeta-5min/](../finetune/2026-07-highbeta-5min/README.md)。

---

## 6. 关键文件指引

| 文件 | 作用 |
|---|---|
| [`/model/kronos.py`](../../model/kronos.py) | Tokenizer + Predictor + KronosPredictor 推理封装 |
| [`/model/module.py`](../../model/module.py) | BSQ 量化器、Transformer 基础模块 |
| [`/finetune/`](../../finetune/) | 本项目微调代码(基于 qlib 多股票) |
| [`/finetune_csv/`](../../finetune_csv/) | 原仓库的单股票 CSV 微调教程 |

---

## 7. 延伸阅读

- 📄 **论文**:Kronos 作者技术报告(见原 [README](../../README.md))
- 🤗 **模型权重**:`NeoQuasar/Kronos-base`、`NeoQuasar/Kronos-Tokenizer-base`
- 🌐 **在线 Demo**:https://shiyu-coder.github.io/Kronos-demo/
