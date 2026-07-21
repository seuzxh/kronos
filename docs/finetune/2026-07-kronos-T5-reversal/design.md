# 设计:实验 5 中期反转策略

> **日期**:2026-07-21
> **前置**:[实验 4](../2026-07-kronos-selected-winsorize/README.md)(精选+Winsorize,完整回测后失败)

---

## 1. 动机:解决 horizon 错配

实验 3/4 失败后,根因分析指向 **horizon 错配**:

```
Kronos 的 range 特征 RankIC 随 horizon:
  T+1:  -0.054
  T+5:  -0.135  ← 2.5 倍
  T+10: -0.160  ← 3 倍
```

Kronos 是**中期反转**信号,但 LGBM 用 T+1 label。强信号被弱 label 稀释。本实验改 label 到 T+5,让 Kronos 的中期优势对齐。

**关键假设**:如果对齐 horizon 后 Kronos 能提升超额收益,说明之前失败是 horizon 问题,不是 Kronos 本身的问题。

## 2. 评估方法学(实验 3/4 的教训)

实验 3 的核心错误:**只看 RankIC 就宣称成功**,没跑完整回测。本实验必须同时报:

| 指标 | 含义 | 必须提升才算成功 |
|---|---|:---:|
| IC | Pearson 线性相关 | 参考 |
| RankIC | Spearman 排序相关 | 参考 |
| **年化超额(扣成本)** | 实际赚钱能力 | ✅ 必须 |
| **夏普** | 风险调整后收益 | ✅ 必须 |
| 最大回撤 | 尾部风险 | 参考 |

**判定标准**(严格):E_T5 的超额 AND 夏普**同时**不低于 A baseline(T+1),才算成功。

## 3. Vintage 重叠回测(核心技术)

### 为什么需要 vintage

qlib 的 `hold_thresh` **只管卖不管买**(Explore 确认,`td0_strategy.py:129`),做不了真正 T+5 持有。改 hold_thresh=5 会导致每天买 + 5 天不卖,累积 ~50 个持仓(5 个重叠 cohort),不是我们想要的。

### Vintage 逻辑

```
每个交易日 T:
  用 pred[T] 选 top-10,形成 vintage V_T
  V_T 持有 5 天(T+1 到 T+5)
基金每日收益 = 当前在跑的 5 个 vintage 的等权平均
换手 = 1/5/天(每天只有 1 个 vintage 换)
```

产出**真日收益序列**,可直接 `risk_analysis(freq="day")`,Sharpe 与 T+1 完全可比。

### 成本模型

- 每天 1 个 vintage 换 = 换手 1/5 = 20%
- 单边成本 0.1%(open 0.05% + close 0.15% 简化)
- 日成本 = (1/5) × 2 × 0.001 = 0.04% = 4 bps

### 指标计算

用 `qlib.contrib.evaluate.risk_analysis(r, freq='day')`,N=238(qlib 约定,非 252)。模式 sum(qlib 默认):年化 = mean × 238,夏普 = mean/std × √238。

## 4. 止损门(关键设计)

在跑 LGBM 前,先验证 **baseline 18 因子在 T+5 是否还有 alpha**。

**逻辑**:如果 baseline(短期动量)在 T+5 已经失效,那即使 Kronos 在中期强,对照线也没意义——整个方向不可行。

**结果**(实际跑出来):
```
baseline 18 因子的 RankIC 衰减(居然不衰减,反而增强):
T+1:  +0.0612  (100%)
T+5:  +0.0959  (157%)  ← 比 T+1 强!
T+10: +0.1020  (166%)
```

**止损门通过**,baseline 在中期有 alpha,继续。

**但这个结果也暗示**:baseline 自己在中期就很强,Kronos 的"中期优势"可能没那么独特。

## 5. 四跑设计

| 跑 | 特征 | Label | hold | 目的 |
|---|---|---|---|---|
| A baseline | 18 因子 | T+1 | 1d | T+1 对照(已有)|
| D 精选 | 21 因子 | T+1 | 1d | 实验 4(已有)|
| **A_T5** | 18 因子 | T+5 | 5d vintage | baseline 在 T+5 |
| **E_T5** | 21 因子(+3 Kronos) | T+5 | 5d vintage | Kronos 在 T+5 |

**全用同一个 vintage 回测**(T+1 的 hold=1),保证可比。

## 6. label 改动(只改 yaml 一行)

```yaml
# A baseline (T+1)
label:
    - Ref($close, -1) / $price_941 - 1

# A_T5 baseline (T+5)
label:
    - Ref($close, -5) / $price_941 - 1
```

Explore 确认:label 只在 yaml 定义,handler/processor 都不读 label。一行改动即可。

## 7. 代码位置

| 组件 | 路径 |
|---|---|
| 止损门诊断 | `finetune/feature_extraction/diag_horizon_decay.py` |
| Vintage 回测 | `finetune/feature_extraction/backtest_vintage.py` |
| T+5 baseline yaml | `3.qlib_ifind_beta/qrun/workflow_kronos_ablation_E_T5_baseline.yaml` |
| T+5 +Kronos yaml | `3.qlib_ifind_beta/qrun/workflow_kronos_ablation_E_T5_selected.yaml` |
