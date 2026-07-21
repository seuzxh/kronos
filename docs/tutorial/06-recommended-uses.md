# 06 — Kronos 怎么用才有效?场景适配指南

> 基于论文实际验证、专家批判性分析、以及我们两次失败实验的一手教训。
>
> 目标:帮你判断"我的场景适不适合用 Kronos",避免重走我们的弯路。

---

## TL;DR

Kronos 是一个**优秀的"K 线形态预测器",但不是"选股 alpha 引擎"**。它的价值在不同场景下差异巨大:

| 场景 | 适用度 | 我们的判断依据 |
|---|---|---|
| 🟢 合成 K 线数据生成 | **强推荐** | 论文核心亮点之一(+22% 保真度),解决数据稀缺 |
| 🟢 波动率预测 | **推荐** | 论文验证(MAE -9%),BSQ 天然编码了 OHLC 范围 |
| 🟢 新市场冷启动预测 | **推荐** | zero-shot 跨市场能力是预训练的核心价值 |
| 🟡 单标的价格/走势预测 | 可尝试 | 论文验证但需谨慎,适合趋势性强的标的 |
| 🟡 异常检测/形态识别 | 可尝试 | BSQ 码本天然支持"罕见形态"识别 |
| 🔴 **A 股横截面选股** | **不推荐** | **我们两次实验证明**(RankIC ≈ 0)|
| 🔴 高频 tick 级策略 | 不推荐 | 用 LOBERT 等订单簿模型更合适 |

---

## 1. 为什么选股不行,但别的可能行?

### 我们的失败根因(回顾)

[03 怎么验证](03-how-we-evaluated.md)详细讲过:Kronos 的训练目标是"预测下一根 K 线 token",这让它擅长**形态层面的预测**,但**横截面选股**需要"比较 100 只股票谁更强"——这是 Kronos 架构里没有的能力。

### 但同样的特性,换个场景就是优势

Kronos 学到的"K 线形态语法"在以下场景**正好是所需**:

```
选股:  需要横截面比较(100 选 10)  → Kronos 缺这个 ❌
合成:  需要生成逼真的 K 线序列      → Kronos 正擅长 ✅
波动率:需要预测 OHLC 范围           → BSQ 编码了这些 ✅
冷启动:需要"没见过也能预测"        → 预训练的核心能力 ✅
```

---

## 2. 三个强推荐场景(详细说明)

### 🟢 场景 A:合成 K 线数据生成(最契合)

**这是什么**:用 Kronos 生成"假的但逼真"的 K 线序列,用于回测、压力测试、数据增强。

**为什么 Kronos 特别合适**:
- 论文报告合成保真度比 DiffusionTS/TimeVAE 高 **22%**
- 能生成包含 103 种经典 K 线形态(头肩顶、十字星等)的序列
- BSQ 码本保证了生成的 K 线在统计分布上接近真实

**实际用法**:
```python
# 伪代码:生成 1000 种"类似 2008 金融危机"的 K 线场景
kronos.generate(
    seed="2008危机的K线模式",
    n_scenarios=1000,
    preserve=["波动率聚集", "厚尾分布", "相关性结构"]
)
# 然后用这些合成数据压力测试你的策略
```

**价值**:解决量化金融最持久的痛点——**历史数据不够用**。你可以测试策略在"百年一遇的危机"中的表现,而不用等历史重演。

**专家观点**:Jonathan Kinlay(25 年量化老兵)认为这是 Kronos **最可信的近期能落地的价值**:
> "The most defensible near-term value is in synthetic data augmentation for stress testing — a workflow enhancement, not a signal source."

### 🟢 场景 B:波动率预测

**这是什么**:预测未来一段时间的价格波动幅度(不是方向)。

**为什么 Kronos 特别合适**:
- K 线的 `high - low` 范围**天然包含波动率信息**
- BSQ 把 OHLC 一起编码,不会丢失这个关系
- 论文验证:波动率 MAE 比基线低 **9%**

**实际用法**:
- 期权定价:波动率是期权价格的核心输入
- 风险管理:预测 VaR(风险价值)
- 仓位管理:波动率高的标的减仓

**对比传统方法**:GARCH(2 个参数,透明)vs Kronos(百万参数,黑盒)。GARCH 在稳定市场够用,Kronos 在复杂/非线性场景可能更优。**建议两者对照,而非替代**。

### 🟢 场景 C:新市场/新资产冷启动

**这是什么**:你要预测一个 Kronos 没专门训练过的市场(如某新兴市场股票、新上市的加密货币)。

**为什么 Kronos 特别合适**:
- 预训练数据来自 **45+ 交易所**,见过各种市场
- zero-shot 能力是预训练的核心卖点:不用任何微调就能给出"合理"预测
- 论文报告:zero-shot 跨市场 RankIC 比通用 TSFM 高 93%

**实际用法**:
- 进入新市场时,先用 Kronos zero-shot 建立**基线预测**
- 传统方法需要数月数据才能建模,Kronos 立即可用
- 随着数据积累,再 fine-tune 到该市场

**价值**:改变了"进入新市场的经济学"——不用等几个月数据才能开始建模。

---

## 3. 如果坚持要用 Kronos 做选股,怎么改?

我们两次实验证明**原生 Kronos 不行**。但理论上可以通过架构改造让它有横截面能力:

### 改造方向 1:加 Ranking Head(中等成本)

```
原 Kronos:  输入 K 线 → 预测下一根 K 线 token
改造后:    输入 K 线 → 预测 K 线 token + 预测相对排名
                                ↑ 新增的头
```

在 predictor 后面加一个"排序头",直接输出"这只股票在所有股票里的相对强弱排名"。这偏离 Kronos 原生设计,等于半个新模型。

### 改造方向 2:横截面注意力(高成本)

让模型能同时看多只股票(如同一行业的全部成分股),学习相对强弱。需要改 self-attention 为 cross-sectional attention。成本高,且没有论文验证。

### 改造方向 3:用作特征提取器(推荐尝试)

**把 Kronos 当"特征提取器",而不是"端到端选股器"**:

```python
# 1. 用 Kronos 预测每只股票的未来 K 线
pred_kline = kronos.predict(stock_history)

# 2. 从预测里提取多个特征(而非只用"预测涨幅")
features = {
    "pred_return": pred_kline.close[-1] / pred_kline.close[0] - 1,
    "pred_volatility": (pred_kline.high - pred_kline.low).mean(),
    "pred_volume_trend": pred_kline.volume.mean() / stock_history.volume.mean(),
    "pred_max_drawdown": ...,
    # ... 更多
}

# 3. 把这些特征喂给传统的横截面模型(LGBM / XGBoost)
final_signal = lightgbm.predict(features_for_all_stocks)
```

**这个思路的妙处**:Kronos 负责"形态理解"(它擅长的),LGBM 负责"横截面比较"(它擅长的)。两者各司其职。

> 💡 这其实是 Foundation Model 在很多领域的标准用法:不端到端,而是当特征提取器。

---

## 4. 微调的最佳实践(如果你确定要微调)

### 4.1 两阶段微调

Kronos 有两个组件,理论上都该微调:

| 阶段 | 组件 | 何时做 | 我们犯的错 |
|---|---|---|---|
| Stage 1 | Tokenizer | 目标域分布与预训练差异大时 | ❌ 我们跳过了(应该做)|
| Stage 2 | Predictor | 总是做 | ✅ 只做了这个 |

**我们的教训**:两次实验都跳过了 tokenizer 微调(为了省时间)。这是 H4 假设,虽然 baseline 失败导致没验证,但理论上是值得试的。

### 4.2 数据量要求

| 频率 | 最少数据量 | 我们的情况 |
|---|---|---|
| 日线 | 2-3 年 | ✅ 2.5 年(够)|
| 1小时 | 6-12 个月 | — |
| 5min | 12+ 个月 | ❌ 我们只有 4 个月(不够)|
| 1min | 6+ 个月 | — |

**我们的教训**:5min 实验只有 4 个月数据,严重不足。数据量不够时,优先降频率。

### 4.3 评估方法(最重要的教训)

**永远不要只用 Val Loss 评估选股任务**。必须用:

```
主指标:RankIC(横截面排序能力)
辅助:累计超额收益、夏普、胜率、IC 衰减
```

详见 [03 怎么验证](03-how-we-evaluated.md)。

### 4.4 何时该放弃 Kronos

如果微调后:
- Val Loss 大幅下降(✅ 形态学到了)BUT
- RankIC 在 0 附近随机(❌ 横截面没学到)

→ **不要继续堆数据/调参**,这是架构限制,不是数据问题。我们两次实验(4 个月数据 vs 2.5 年数据)已经证明数据量解决不了。

---

## 5. 决策流程图

```
你的任务是什么?
│
├─ 合成数据生成 / 压力测试
│  → ✅ 直接用 Kronos,最契合
│
├─ 波动率预测(期权/风控)
│  → ✅ 推荐,对照 GARCH 看哪个好
│
├─ 新市场冷启动预测
│  → ✅ zero-shot 为主,数据够了再微调
│
├─ 单标的价格预测(如 BTC)
│  → 🟡 可尝试,关注趋势性强的标的
│
├─ 选股(横截面)
│  → 🔴 别端到端用 Kronos
│  → 考虑"Kronos 特征提取 + LGBM 排序"
│  → 或直接用因子模型(LGBM IC 0.067 已验证)
│
└─ 高频/tick 级
   → 🔴 用 LOBERT 等订单簿模型,不用 Kronos
```

---

## 6. 常见误区

### 误区 1:"Kronos RankIC 比基线高 93%,所以能选股"

**真相**:论文的 93% 是**相对提升**(比最差基线高 93%),不是绝对值。绝对 RankIC 可能只有 0.02~0.03(弱 alpha)。而且论文的 benchmark 不等于 A 股实盘。

### 误区 2:"微调后 Val Loss 降了很多,模型学会了"

**真相**:Val Loss 衡量"形态重建",不衡量"选股能力"。我们实验 2 Val Loss 降 42%,但 RankIC 仍是 0。详见 [03 怎么验证](03-how-we-evaluated.md)。

### 误区 3:"数据越多效果越好"

**真相**:如果方法/架构不对,堆数据没用。我们 5min(4 月)→ 日频(2.5 年),数据增 7 倍,RankIC 没改善。

### 误区 4:"开源了就能直接生产用"

**真相**:作者明确声明研究用途。生产化需要组合优化、风险中性化、成本滑点建模等大量额外工程。Kronos 只是其中一环。

---

## 7. 推荐的学习/实践路径

按优先级排序:

### 初级(1-2 周)
1. **跑通 Kronos zero-shot 预测**(官方 examples):理解它的输入输出
2. **复现我们的 RankIC 评估**(本仓库代码):理解为什么"loss 低 ≠ 能赚钱"
3. **试一次合成数据生成**:感受 Kronos 最强的能力

### 中级(2-4 周)
4. **在一个新市场(如加密货币)试 zero-shot**:验证跨市场能力
5. **对照 GARCH 做波动率预测**:看 Kronos 是否更优
6. **试"Kronos 特征 + LGBM"的混合方案**:这是我们没做但推荐的路径

### 高级(1-2 月)
7. **fine-tune tokenizer + predictor**:在我们失败的 baseline 上继续(如果能拿到 tokenizer 微调收益)
8. **研究 Ranking Head 改造**:让 Kronos 有横截面能力(高研究价值)
9. **用合成数据做策略压力测试**:把 Kronos 的强项用到实处

---

## 8. 参考资源

### 论文与官方
- [Kronos 论文 arXiv](https://arxiv.org/abs/2508.02739)(AAAI 2026）
- [官方 GitHub](https://github.com/shiyu-coder/Kronos)
- [HuggingFace 模型](https://huggingface.co/NeoQuasar/Kronos-base)

### 批判性分析(强烈推荐)
- **[Jonathan Kinlay: Time Series Foundation Models for Financial Markets](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)** — 25 年量化老兵的冷静分析,本文"3 个误区"和"波动率对照"深受他启发
- **[Pebblous: Kronos Domain Deep-Dive](https://blog.pebblous.ai/report/kronos-financial-foundation-model/en/)** — 详细的技术与生态分析,含与 25 个基线的对比

### 我们的一手实验记录
- [实验 1:高贝塔 5min](../finetune/2026-07-highbeta-5min/README.md) ❌
- [实验 2:高贝塔指增日频](../finetune/2026-07-highbeta-enhancement/README.md) ❌
- [为什么 loss 低 ≠ 能赚钱](03-how-we-evaluated.md)

---

## 风险提示

- 本指南基于**当前时点**(2026-07)的信息,模型和生态会演化
- 我们的失败结论限于 **A 股高贝塔选股**,其他场景需独立验证
- 任何量化策略都需考虑交易成本、滑点、容量,**不构成投资建议**
