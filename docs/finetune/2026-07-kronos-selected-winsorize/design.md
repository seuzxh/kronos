# 设计:实验 4 Kronos 特征精选 + Winsorize

> **日期**:2026-07-21
> **前置**:[实验 3](../2026-07-kronos-5min-features-lgbm/README.md)(Kronos 特征提取器,RankIC +6.4%)

---

## 1. 动机:为什么做精选 + Winsorize

实验 3 加了 8 个 Kronos 特征后,LGBM 的 OOS RankIC 从 0.0612 提升到 0.0651(+6.4%)。但诊断发现两个问题:

### 问题 1:8 个特征里只有 3 个真有信号

单特征 RankIC 分析(15,000 样本采样,T+1 horizon):

| 特征 | RankIC | 信号强度 |
|---|:---:|---|
| `range` | -0.054 | 强 |
| `low_ratio` | +0.035 | 中强 |
| `uncertainty` | -0.033 | 中强 |
| `high_ratio` | -0.033 | 中(与 range 冗余,corr -0.85)|
| `afternoon_ret` | +0.014 | 弱噪声 |
| `ret_intraday` | +0.008 | 噪声(与 morning_ret 冗余 0.88)|
| `morning_ret` | +0.005 | 噪声 |
| `close_pos` | +0.004 | 噪声 |

**8 个特征的有效维度只有 ~3 个**,其余 5 个要么是噪声,要么与强特征高度冗余。

### 问题 2:异常值扰动 IC

C 跑出现 IC 下降 10% 但 RankIC 上升的反直觉现象:

```
A baseline:  IC=0.0548  RankIC=0.0612
C 全 Kronos: IC=0.0491  RankIC=0.0651
             ↑ IC 下降   ↑ RankIC 上升
```

这说明 Kronos 特征有异常值,扰动了 Pearson IC(线性相关),但改善 Spearman RankIC(排序)。选股只关心排序,所以 RankIC 是正确指标。但 IC 下降仍是个隐患 —— 极端异常值可能让 LGBM 的分裂点偏移。

### 假设

**假设 1**:删掉 5 个噪声/冗余特征后,LGBM 学得更干净,RankIC 可能进一步提升(噪声特征不再分散模型注意力)。

**假设 2**:对 3 个保留特征做 Winsorize(1%/99% 截尾),去除异常值,IC 应该恢复到 baseline 附近,RankIC 可能小幅提升。

本实验同时验证这两个假设。

---

## 2. 设计

### 2.1 精选 3 个 Kronos 特征

保留:
- `kronos_range` —— 最强反转信号(T+1 RankIC -0.054,T+10 RankIC -0.160)
- `kronos_low_ratio` —— 反转后动量(T+1 +0.035,T+10 +0.095)
- `kronos_uncertainty` —— 模型信心度(T+1 -0.033,T+10 -0.120)

删掉:`high_ratio`(与 range 冗余)、`afternoon_ret/ret_intraday/morning_ret/close_pos`(弱噪声)。

实现:新建 `MinuteEnhancedKronosSelectedHandler`,21 因子 = 18 baseline + 3 精选。

### 2.2 Winsorize 只作用于 3 个 Kronos 字段

**scope 决策**(已确认):只 Winsorize 3 个 Kronos 特征,**不动现有 18 因子 baseline**。理由:
- 18 因子已在生产验证(Winsorize 它们可能破坏 baseline)
- D vs C 的对比要纯净:差异只来自"精选 + Winsorize Kronos"

实现:自写 `Winsorize` Processor(qlib 没有内置),放在 `qlib_ifind_beta/processors.py`。

```python
class Winsorize(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group="feature",
                 fields_subset=None, q_lower=0.01, q_upper=0.99):
        ...
    def fit(self, df):
        # 只在训练段算分位数(防泄露)
        df_train = fetch_df_by_index(df, [fit_start, fit_end], level="datetime")
        self.lo = np.nanquantile(df_train[cols], 0.01, axis=0)
        self.hi = np.nanquantile(df_train[cols], 0.99, axis=0)
    def __call__(self, df):
        # 全时段 clip(用训练段分位数)
        df[cols] = np.clip(df[cols], self.lo, self.hi)
```

关键:
- `fit_start_time/fit_end_time` 由 qlib 的 `check_transform_proc` 自动注入(从 yaml 的 `fit_start_time: 2024-01-01, fit_end_time: 2025-12-31`),**无前视泄露**
- 放 `learn_processors`(训练段 fit),不放 `shared_processors`(避免与 L1 前视护栏冲突)
- `fields_subset` 支持,可只 Winsorize 指定字段

### 2.3 对比设计

| 跑 | 特征数 | Winsorize | 对比目的 |
|---|:---:|:---:|---|
| A baseline | 18 | — | 对照(已有 RankIC 0.0612)|
| C 全 Kronos | 26 | — | 实验 3 结果(已有 RankIC 0.0651)|
| **D 精选+Winsorize** | 21 | 3 字段 | **新跑** |

时间切分、模型(HFLGBModel binary loss)、策略(topk=10, n_drop=8)、label(`Ref($close,-1)/$price_941 - 1`)**全 FROZEN**,唯一变量是特征集 + Winsorize。

复用实验 3 的 bin(已写好,finetune 版),**不用重新特征提取**。

---

## 3. 判定标准

| 结果 | 含义 | 后续 |
|---|---|---|
| D RankIC > C + 5% | 精选有效,噪声拖后腿 | 继续优化(中期反转方向)|
| D RankIC ≈ C | Kronos 天花板,精选/Winsorize 不改变边界 | 转向中期反转 |
| D RankIC < C | LGBM 能从冗余特征提取信息 | 保留全量,不精选 |

---

## 4. 代码位置

| 组件 | 路径 |
|---|---|
| Winsorize processor | `3.qlib_ifind_beta/qlib_ifind_beta/processors.py` |
| 精选 handler | `3.qlib_ifind_beta/qlib_ifind_beta/minute_enhanced_handler.py:MinuteEnhancedKronosSelectedHandler` |
| D 跑 yaml | `3.qlib_ifind_beta/qrun/workflow_kronos_ablation_D_selected.yaml` |
