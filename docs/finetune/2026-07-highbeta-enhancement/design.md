# 方案设计 - 高贝塔指增 Kronos 微调

> 本文档讲清楚"**为什么这么设计**":指增思路如何转化为 Kronos 可优化的目标、超额收益预测的架构、与上次方案的差异。
> 工程实施细节见各 Loop 文档。

---

## TL;DR

- **核心创新**:把 Kronos 从"预测个股 K 线"重塑为"预测个股**相对指数**的 K 线",信号=预测超额收益排名。
- **4 个可证伪假设**(Loop 第②阶段):每个假设都有明确的证伪条件,避免盲目迭代。
- **不改动 Kronos 模型架构**,只改**输入特征**(加指数作为条件)+ **评估指标**(超额 RankIC)。

---

## 1. 指增思路的本质

指数增强(Enhancement)的目标:**跑赢基准指数**。数学表达:

```
超额收益(股票 i, 第 t 天) = 个股收益(股票 i, t) − 指数收益(t)
                           = R_i,t − R_index,t

策略收益 = Σ (选股超额收益) − 交易成本
```

**关键洞察**:传统 Kronos 预测 `R_i,t`(绝对收益),但个股收益 ≈ β·R_index + α_i。高贝塔股 β 大,模型若只学到 β(市场涨跌),会有"高 Val Loss 改善但无 alpha"的假象(上次实验的失败模式)。

**本方案**:让模型预测 `α_i,t = R_i,t − β_i·R_index,t`(或简化为 `R_i,t − R_index,t`),直接剥离市场 beta,强迫模型学 alpha。

---

## 2. 三种"超额收益"的实现选择(已决策)

### 选项 A:预测个股 K 线,信号用**预测超额收益**排名 ✅ **采用**

```
Kronos 输入: 个股历史 OHLCV + 指数历史 OHLCV(作为额外特征通道)
Kronos 输出: 个股未来 K 线 token
信号计算:    预测涨幅_top − 预测指数涨幅 = 预测超额涨幅
选股:        预测超额涨幅 top-K 等权
```

**优点**:
- 完全兼容 Kronos 原生架构(预测 K 线 token),不改模型
- 指数作为条件输入,模型可以学到"相对强弱"
- 信号层面直接体现指增本质

**为何不用选项 B/C**:
- **选项 B(改 ranking head)**:偏离 Kronos 原生架构,等于换模型,违背"先验证再微调"原则
- **选项 C(预测纯 alpha 序列)**:失去 K 线形态信息,Kronos 的 tokenizer 优势无法发挥

> 📝 这个决策回应了上次 [backtest-report.md](../2026-07-highbeta-5min/backtest-report.md) 改进方向 #1/#2(改预测目标、加横截面信息),但用**最小改动**实现。

---

## 3. 架构设计

### 3.1 输入特征(关键改动)

```
原 Kronos 输入(6 维):  [open, high, low, close, vol, amt]  ← 个股自身
本方案输入(12 维):      [个股 6 维 OHLCV] + [指数 6 维 OHLCV]
                                                 ↑ 新增
```

**指数特征如何对齐**:
- 指数 883926.TI 与个股同一时间轴(交易日)
- 指数特征做**与个股相同**的窗口 z-score 归一化(用 lookback 段统计,防泄露)
- 指数 vol/amt 用指数自身的(非加权聚合)

### 3.2 配置参数

| 参数 | 日频 Loop | 分钟 Loop |
|---|---|---|
| lookback | 240 个交易日(约 1 年)| 240 根 5min(5 交易日,沿用上次)|
| predict | 5 个交易日 | 48 根(次日全天,沿用上次)|
| 特征维度 | 12(个股 6 + 指数 6)| 12 |
| max_context | 512(240+5+1=246 < 512 ✅)| 512(240+48+1=289 ✅)|
| CV 切分 | 4 折扩张式(6 年数据充足)| 4 折扩张式(仅 4 个月,见门禁)|

> ⚠️ 日频 lookback=240 (1 年) 比 5min lookback=240 (5 天) 信息量大得多,这是日频先行的另一个理由。

### 3.3 信号与选股(核心改变)

```python
# 上次(失败):预测涨幅 top-10
signal = predicted_close[-1] / predicted_close[0] - 1
selected = top_10(signal)

# 本次:预测超额涨幅 top-K
stock_signal = predicted_stock_return          # 个股预测收益
index_signal = predicted_index_return          # 指数预测收益(同一模型,指数作为输入)
excess_signal = stock_signal - index_signal    # 超额收益
selected = top_K(excess_signal, K=10)
```

**实现细节**:同一 Kronos 模型对每只个股预测时,输入包含该股 + 指数的历史;同时对**指数本身**做一次预测(输入纯指数历史),得到 `predicted_index_return`。两者相减得超额。

### 3.4 评估指标(决定 Loop 方向)

| 指标 | 公式 | 目标 | 用途 |
|---|---|---|---|
| **超额 RankIC** | Spearman(预测超额排名, 实际超额排名) | > 0.03 | 主指标,门禁 |
| 超额累计收益 | Σ(top-K 超额) | > 0 | 策略层 |
| 超额夏普 | mean(超额) / std(超额) × √252 | > 0.5 | 风险调整 |
| 胜率 | 超额 > 0 的天数占比 | > 52% | 稳定性 |
| IC 衰减 | lag-N 的 RankIC | 缓慢下降 | 信号有效期 |
| 个股 Val Loss | token 重建误差 | 下降 | 健康度(辅助)|

> 💡 **关键设计**:超额 RankIC 是**主指标**,个股 Val Loss 只是辅助。这直接修正了上次"只盯 Val Loss"的错误。

---

## 4. 4 个可证伪假设(Loop 第②阶段)

每个假设都有**明确的证伪条件**,失败就放弃该方向,不盲目迭代。

### 假设 H1:加指数特征能提升横截面区分度
- **预测**:超额 RankIC 比"无指数特征"基线高 ≥ 20%
- **证伪**:跑 2 折对比实验,RankIC 无显著差异或反向
- **如果证伪**:说明指数特征对模型无用,改考虑行业/风格因子

### 假设 H2:超额收益信号比绝对收益信号更赚钱
- **预测**:同模型同参数,超额信号累计收益 > 绝对信号
- **证伪**:4 折回测超额信号累计收益 ≤ 绝对信号
- **如果证伪**:回到绝对信号,但保留超额 RankIC 作为诊断指标

### 假设 H3:日频比 5min 更适合此任务
- **预测**:日频超额 RankIC > 5min(同样 4 折)
- **证伪**:日频 RankIC 无优势
- **如果证伪**:说明频率不是瓶颈,聚焦其他改进

### 假设 H4:微调 tokenizer 能改善 alpha
- **预测**:微调 tokenizer 后超额 RankIC 比复用预训练高 ≥ 10%
- **证伪**:两阶段微调 RankIC 无改善
- **如果证伪**:tokenizer 不是瓶颈,聚焦 predictor 或特征

> 📝 这 4 个假设覆盖了上次 [backtest-report.md](../2026-07-highbeta-5min/backtest-report.md) 改进方向 #1-#5,每个都是可独立验证的单元。

---

## 5. 不做什么(边界)

为避免范围蔓延,明确**不做**:

1. **不改 Kronos 模型架构**(不加 ranking head,不改 tokenizer 算法)
2. **不做风险因子中性化**(规模/价值/动量等 Barra 风格因子)— 生产化阶段才做
3. **不做组合优化**(Markowitz / 风险预算)— 信号验证阶段先用等权
4. **不做高频回测**(tick 级 / L2 数据)— 分钟级已是上限
5. **不做实盘对接**(券商 API / 风控)— 本方案止于离线回测

---

## 6. 代码改动清单(预估)

| 文件 | 类型 | 改动 |
|---|---|---|
| `finetune/enhancement/config_daily.py` | 新增 | 日频 Loop 配置 |
| `finetune/enhancement/config_minute.py` | 新增 | 分钟 Loop 配置 |
| `finetune/enhancement/index_loader.py` | 新增 | 加载 883926.TI 指数行情 |
| `finetune/enhancement/preprocess_daily.py` | 新增 | 日频预处理(个股+指数对齐)|
| `finetune/enhancement/dataset_enh.py` | 新增 | 12 维特征 Dataset(继承 HighbetaDataset)|
| `finetune/enhancement/backtest_excess.py` | 新增 | 超额收益回测 + RankIC |
| `finetune/enhancement/run_daily.sh` | 新增 | 日频 Loop 编排 |
| `finetune/enhancement/run_minute.sh` | 新增 | 分钟 Loop 编排(含门禁检查)|
| `finetune/enhancement/fetch_data.sh` | 新增 | iFinD 拉股池+指数 |
| `finetune/dataset.py` | 复用 | 归一化/窗口采样逻辑 |

详细实施见各 Loop 文档。

---

## 7. 为什么相信这次会不一样?

诚实评估:无法保证赚钱。但本方案**系统性地消除了上次失败的根因**:

| 上次失败根因 | 本方案应对 | 是否彻底解决 |
|---|---|---|
| 只预测形态,无横截面 alpha | 预测超额收益,直接优化横截面 | 部分(仍受限于 Kronos 能力)|
| 评估只看 Val Loss | 主指标改为超额 RankIC | ✅ 彻底 |
| 5min 数据仅 4 个月 | 日频 6 年优先,分钟严格门禁 | ✅ 彻底 |
| Tokenizer 复用预训练 | 作为 H4 假设显式验证 | 待验证 |
| 信号口径 bug(v1/v2) | 信号计算单元测试 + 双口径对照 | ✅ 彻底 |

**结论**:本方案比上次有显著改进,但金融预测的内在不确定性仍在。Loop 工程化的价值在于**快速证伪假设、找到真实能力边界**,而不是承诺赚钱。
