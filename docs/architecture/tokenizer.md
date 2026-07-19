# Kronos Tokenizer 原理

> 深入理解"K 线怎么变成 token"。前置阅读:[kronos-overview.md](kronos-overview.md)
> 代码:`/model/kronos.py` 的 `KronosTokenizer` 类 + `/model/module.py` 的 `BSQuantizer` 类

---

## TL;DR

- Tokenizer 的任务:把每根 K 线(6 维连续特征)**编码成 2 个整数 token**(`s1` 12 bit + `s2` 4 bit),且**可解码回 K 线**。
- 核心技术:**BSQ(Binary Spherical Quantizer)** —— 把连续向量投影到超球面,再二值化成 ±1 串作为离散码。
- 训练目标:重建 K 线(MSE)+ 码本利用率正则(避免码本坍缩)。
- 代码入口:`tokenizer.encode(x)` → token,`tokenizer.decode(token)` → K 线。

---

## 1. 为什么要 tokenize K 线?

LLM 处理的是离散 token(词)。要把同样的自回归框架搬到 K 线,必须先把连续 K 线**离散化**。

要求:
1. **可逆**:能从 token 解码回 K 线(否则无法生成预测结果)
2. **语义稳定**:相似的 K 线应有相近/相同的 token(模型好学)
3. **码本均衡**:不能所有 K 线都坍缩到少数几个 token(否则信息丢失)

Kronos 用 **VQ-VAE 风格的 Encoder-Decoder + BSQ 量化器** 实现这些目标。

---

## 2. 整体结构

```
   K 线 x (B, T, d_in=6)
        │
        ▼  Linear(d_in → d_model)
   ┌────────────────────────┐
   │ Encoder Transformer    │  ×(n_enc_layers-1)
   │ (自注意力提取时序特征)  │
   └────────────────────────┘
        │
        ▼  Linear(d_model → s1_bits + s2_bits)   # e.g. 12+4=16 维
   ┌────────────────────────┐
   │  BSQ 量化器            │
   │  连续向量 → 离散 token  │   ← 核心创新点
   └────────────────────────┘
        │
   z_indices (B, T, 2)  ← 每根 K 线得到 (s1_id, s2_id)
        │
        ▼  Linear → Decoder Transformer ×(n_dec_layers-1) → Linear
   ┌────────────────────────┐
   │ Decoder (重建 K 线)    │
   └────────────────────────┘
        │
        ▼
   x_hat (B, T, d_in)   ← 重建 K 线,与输入算 MSE
```

**双 token 设计**(`HierarchicalEmbedding` + `DualHead`):
- `s1`(12 bit = 4096 个码):粗粒度主语义(K 线的大类形态)
- `s2`(4 bit = 16 个码):细粒度残差(同 s1 类内的细微差异)
- Predictor 分别预测 s1 和 s2 两个头

---

## 3. BSQ 量化器(核心)

> 论文:https://arxiv.org/pdf/2406.07548.pdf

### 3.1 量化过程(`quantize` 方法)

输入:`z` 是 BSQ 前的连续向量,维度 = `embed_dim`(如 12)。

```python
# 关键 1 行:把每个分量二值化
zhat = where(z > 0, +1, -1)
# 直通估计器(Straight-Through Estimator):前向用 zhat,反向梯度传给 z
zq = z + (zhat - z).detach()
```

每个分量取 ±1,16 维向量就有 `2^12 × 2^4 = 65536` 种可能(对应 s1/s2 组合)。
然后用 `codes_to_indexes` 把 ±1 串转成一个整数索引(token id)。

**为什么用 ±1 二值而不是 k-means 码本(VQ)?**
- 不需要学码本嵌入(每维只有 2 个值,码本隐式确定)
- L2 归一化后落在单位球面上,数值稳定
- 训练更稳,不易码本坍缩

### 3.2 球面归一化

```python
q_scale = 1.0 / sqrt(embed_dim)   # 把 ±1 串缩到单位球面
zq = zq * q_scale
```

归一化后所有码字在单位超球面上等距分布,避免某些维度主导。

### 3.3 三个损失项(`forward` 方法)

```python
# (1) Commit loss:让 encoder 输出 z 靠近量化点 zq
commit_loss = beta * mean((zq.detach() - z)^2)

# (2) Entropy penalty:让码本利用率均衡(关键!)
#     persample_entropy 高 = 每个 sample 的码分布不确定(好)
#     cb_entropy 低 = 码本整体使用均衡(好)
entropy_penalty = gamma0 * persample_entropy - gamma * cb_entropy

# (3) 重建 loss(在 Tokenizer 外层算 MSE)
```

**熵正则的作用**:防止码本坍缩(只用到少数几个码)。`gamma0 * H_sample - gamma * H_codebook` 这个组合鼓励:
- 每个 sample 概率分散(用 soft entropy 近似)
- 码本整体使用均匀

> 💡 这就是 BSQ 比 VQ-VAE 强的地方:VQ 用 commitment loss 容易死码,BSQ 用熵正则更稳。

### 3.4 分组近似(`group_size`)

完整码本熵 `H_codebook` 在 `2^16` 个码上算太贵。BSQ 把 16 维分成 `16/9 ≈ 2` 组,只在每组 `2^9` 上算熵再求和作为近似。

代码里 `soft_entropy_loss` 方法实现了这个分组 softmax 熵。

---

## 4. 训练 vs 推理

### 训练时(单独训练 tokenizer)

```bash
python finetune/train_tokenizer.py   # 用 K 线数据训练 tokenizer
```

Loss = MSE(重建) + BSQ loss(commit + entropy)。目标:让 tokenizer 学会"用 token 准确重建 K 线"。

### 推理时(微调 predictor 时)

Tokenizer **冻结**,只作为编码/解码工具:
- `encode(x)`:K 线 → token(给 predictor 当 target)
- `decode(token)`:predictor 输出的 token → K 线(生成预测结果)

> ⚠️ **本项目的关键决策**:高贝塔 5min 微调时**复用预训练 tokenizer,没单独训练**。这是回测不赚钱的疑点之一(见 [改进方向](../finetune/2026-07-highbeta-5min/README.md#改进方向))—— 5min 分布可能和 tokenizer 训练数据不一致。

---

## 5. 关键超参(本项目 base 配置)

| 参数 | 值 | 含义 |
|---|---|---|
| `d_in` | 6 | 输入特征维度(OHLCV+amount)|
| `d_model` | see config | Transformer 隐层维度 |
| `s1_bits` | 12 | 主 token 位数 → 4096 个码 |
| `s2_bits` | 4 | 残差 token 位数 → 16 个码 |
| `group_size` | 9 | 熵估计的分组大小 |
| `beta` / `gamma0` / `gamma` / `zeta` | see config | 各 loss 项权重 |

---

## 6. 验证 tokenizer 是否学得好

可以从两个角度验证:

1. **码本利用率**:统计训练集所有 K 线的 token 分布,看熵是否接近最大熵(说明码本用得均衡)
2. **重建误差**:对一批 K 线 `encode → decode`,算 MSE,看是否能准确还原

本项目 `finetune/check_data.py` 中有 OHLC 一致性检查(open ≤ high、low ≤ close 等),间接验证 token 解码合理性。

---

## 7. 延伸阅读

- 🔬 BSQ 论文:https://arxiv.org/pdf/2406.07548.pdf
- 🔬 VQ-VAE 基础:https://arxiv.org/abs/1711.00937
- 📁 代码:[`/model/kronos.py`](../../model/kronos.py) `KronosTokenizer` 类(40-178 行)
- 📁 代码:[`/model/module.py`](../../model/module.py) `BinarySphericalQuantizer` 类(39-223 行)
