# Kronos Predictor 原理

> 理解"给定历史 token,模型怎么预测未来"。前置阅读:[kronos-overview.md](kronos-overview.md) / [tokenizer.md](tokenizer.md)
> 代码:`/model/kronos.py` 的 `Kronos` 类(predictor 模型)+ `KronosPredictor` 类(推理封装)

---

## TL;DR

- Predictor 是 **GPT 风格的 decoder-only Transformer**,在 token 空间做自回归预测。
- 双 token 结构 → **双头预测**:先预测 `s1`(主语义),再**条件于 s1** 预测 `s2`(残差)。
- 训练:s1/s2 双头交叉熵;推理:温度采样 + top-p,支持并行多采样。
- 代码入口:`KronosPredictor.predict(df, ...)` 一行完成"K 线 → 预测 K 线"。

---

## 1. 任务定义

给定历史 K 线 token 序列 `[t_1, t_2, ..., t_L]`,预测未来 token `[t_{L+1}, ..., t_{L+P}]`。

其中每个 `t_i` 实际是 `(s1_i, s2_i)` 一对 token(见 [tokenizer.md](tokenizer.md))。

这和 GPT 预测下一个词完全同构,只是"词"换成了"K 线 token"。

---

## 2. 模型结构(`Kronos` 类)

```
   输入:s1_ids, s2_ids (B, T)   + 时间戳 stamp (B, T, 5)
        │
        ▼
   ┌──────────────────────────────┐
   │ HierarchicalEmbedding        │  s1/s2 各自查表后融合
   │   + TemporalEmbedding        │  + 时间特征(分钟/时/周/日/月)
   └──────────────────────────────┘
        │
        ▼  token_dropout
   ┌──────────────────────────────┐
   │ TransformerBlock × n_layers  │  causal self-attention + RoPE
   │ (decoder-only, 因果掩码)     │
   └──────────────────────────────┘
        │
        ▼  RMSNorm
   ┌──────────────────────────────┐
   │ DualHead → s1_logits         │  (B, T, 2^s1_bits)
   └──────────────────────────────┘
        │
        │  从 s1_logits 采样得到 s1_pred
        ▼
   ┌──────────────────────────────┐
   │ DependencyAwareLayer         │  把 s1_pred 的 embedding 注入隐状态
   │   + DualHead.cond_forward    │
   │   → s2_logits                │  (B, T, 2^s2_bits)   ← 条件于 s1
   └──────────────────────────────┘
```

### 关键设计点

#### (1) 分层嵌入 `HierarchicalEmbedding`
s1(12bit,4096 码)和 s2(4bit,16 码)各自有独立的 embedding 表,相加融合。s1 携带主语义,s2 携带残差。

#### (2) 时间嵌入 `TemporalEmbedding`
K 线是时序数据,时间戳信息(分钟、小时、星期、日、月)很重要。和 token embedding 相加注入。

#### (3) 因果自注意力 + RoPE
标准 GPT 配方:causal mask 保证只看历史,RoPE(Rotary Positional Embedding)注入相对位置。

#### (4) DependencyAwareLayer(s2 条件于 s1)
**核心创新**:s2 不是独立预测的,而是**先预测 s1,再用 s1 的 embedding 作为条件预测 s2**。这符合"先确定大类形态,再细化残差"的层次结构。

- 训练时用 `use_teacher_forcing=True`:用 ground-truth s1 当条件(稳定)
- 推理时从 s1_logits 采样 s1,再用它当条件

---

## 3. 训练目标(`DualHead.compute_loss`)

```python
loss = s1_loss + s2_loss
# s1_loss = CrossEntropy(s1_logits,  s1_targets)
# s2_loss = CrossEntropy(s2_logits,  s2_targets)   # 条件于 s1
```

⚠️ **关键理解**(本项目踩的坑,见 [backtest-report](../finetune/2026-07-highbeta-5min/backtest-report.md)):

> Val Loss 衡量的是 **token 重建误差**(模型学会了"预测 5min K 线形态"),
> **不等于** 横截面选股 alpha(区分哪些股票明天涨更多)。
> 这是两个不同任务。Val Loss 低不代表能赚钱。

---

## 4. 推理流程(`KronosPredictor.predict`)

`KronosPredictor` 是推理封装,一行调用完成端到端预测:

```python
predictor = KronosPredictor(model, tokenizer, device='cuda', max_context=512, clip=5)
df_pred = predictor.predict(
    df,                       # 历史 K 线 DataFrame
    x_timestamp=...,          # 输入时间戳
    y_timestamp=...,          # 要预测的时间戳
    pred_len=48,              # 预测长度
    T=1.0,                    # 采样温度
    top_k=0,                  # top-k 采样,0=不启用
    top_p=0.9,                # nucleus sampling
    sample_count=20,          # 采样次数,取均值得概率预测
)
```

### 推理步骤

```
1. 输入 df 做 z-score 归一化(只用 lookback 段,防未来泄露)
2. tokenizer.encode → token 序列
3. 自回归生成 pred_len 个新 token:
   for i in range(pred_len):
       logits = model.forward(history_tokens)
       next_token = sample(logits[-1], T, top_k, top_p)
       history_tokens.append(next_token)
4. tokenizer.decode(生成的 token) → 归一化 K 线
5. 反归一化 → 真实预测 K 线
6. 若 sample_count > 1,重复 1-5,取均值
```

### 采样参数影响

| 参数 | 作用 | 典型值 |
|---|---|---|
| `T` (温度) | T 大→更多样,T 小→更确定 | 1.0 标准;<1 保守;>1 随机 |
| `top_p` | nucleus sampling 阈值 | 0.9(保留前 90% 概率质量)|
| `sample_count` | 采样次数,取均值得概率预测 | 1(单点)/ 20(平滑)|

> 💡 本项目高贝塔回测用 `sample_count=20` 取均值,可能过度平滑,是改进方向之一。

---

## 5. 关键超参(base 配置)

| 参数 | 值 | 含义 |
|---|---|---|
| `n_layers` | see config | Transformer 层数 |
| `d_model` | see config | 隐层维度 |
| `n_heads` | see config | 注意力头数 |
| `s1_bits` / `s2_bits` | 12 / 4 | 与 tokenizer 对齐 |
| `learn_te` | bool | 时间嵌入是否可学 |
| `max_context` | 512 | 最大序列长度(限制 lookback+predict)|
| `clip` | 5 | 归一化后值域裁剪(防异常值)|

---

## 6. 微调 vs 预训练

### 预训练(原作者已完成)
在全球 45+ 交易所数据上大规模训练 tokenizer + predictor,得到 HuggingFace 上的 `NeoQuasar/Kronos-base`。

### 微调(本项目做的事)
冻结 tokenizer,只在下游数据(如本项目高贝塔股池 5min)上**继续训练 predictor**:

```python
# 伪代码:finetune/train_predictor.py 核心循环
for epoch in range(8):
    for batch in dataloader:
        x, stamp = batch              # 历史 K 线 + 时间戳
        s1_ids, s2_ids = tokenizer.encode(x)   # 冻结,只前向
        s1_logits, s2_logits = model(s1_ids, s2_ids, stamp)
        loss = dual_head.compute_loss(s1_logits, s2_logits, s1_ids, s2_ids)
        loss.backward()
        optimizer.step()
```

详见 [finetune-pipeline.md](finetune-pipeline.md)。

---

## 7. 延伸阅读

- 📁 代码:[`/model/kronos.py`](../../model/kronos.py)
  - `Kronos` 类(180-310 行):模型本体
  - `KronosPredictor` 类(482-620 行):推理封装
- 📁 代码:[`/model/module.py`](../../model/module.py)
  - `HierarchicalEmbedding` / `DualHead` / `DependencyAwareLayer`:层次化预测组件
  - `TransformerBlock` / `RoPE`:标准 Transformer 模块
- 📁 训练:[`/finetune/train_predictor.py`](../../finetune/train_predictor.py)
