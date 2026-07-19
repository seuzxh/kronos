# Stage 1 - 日频 Loop 详细设计

> 这是当前主战场。日频数据 6 年充足,先在这里验证指增思路是否 work。
> 设计依据见 [design.md](design.md)。

---

## TL;DR

- **目标**:日频超额 RankIC > 0.03(弱 alpha 起点),持续 2 折
- **窗口**:lookback=240 交易日(~1 年)→ predict=5 交易日
- **特征**:12 维(个股 6 + 指数 6)
- **CV**:4 折扩张式(2020-03 ~ 2026-05)
- **门禁**:超额 RankIC > 0.03 持续 2 折 → 解锁 Stage 2 分钟 Loop

---

## ① 目标定义(Goal)

### 主指标(决定门禁)

| 指标 | 目标 | 测量方式 |
|---|---|---|
| **日频超额 RankIC** | > 0.03 | 每日 Spearman(预测超额排名, 实际超额排名),折内取均值 |

### 辅助指标(诊断用)

| 指标 | 目标 | 说明 |
|---|---|---|
| 超额累计收益 | > 0(4 折合并)| 不亏钱是底线 |
| 超额夏普 | > 0.5 | 风险调整后仍有 alpha |
| 胜率 | > 52% | 选股方向正确率 |
| IC 衰减(lag-5) | > 0.5×lag-0 IC | 信号至少 5 天有效 |
| 个股 Val Loss | < zero-shot 基线 | 微调生效(只是健康度)|

> ⚠️ 上次错误的根因:把 Val Loss 当主指标。本次主指标明确是**超额 RankIC**。

---

## ② 假设生成(Hypothesis)

按优先级排序,每个假设对应一轮实验。**单变量原则**:每轮只改一个变量。

### H1:加指数特征能提升横截面区分度(本轮首选)
- **实验**:对比"12 维(个股+指数)" vs "6 维(纯个股)" 基线
- **预期**:超额 RankIC 提升 ≥ 20%
- **证伪**:2 折对比无显著差异
- **成本**:1 次预处理 + 2 次训练

### H2:超额收益信号 > 绝对收益信号
- **实验**:同模型,信号分别用 `pred_stock - pred_index` vs `pred_stock`
- **预期**:超额信号累计收益更高
- **依赖**:H1 通过(否则无意义)

### H3:微调 tokenizer 改善 alpha
- **实验**:两阶段(tokenizer + predictor)vs 单阶段(predictor only)
- **预期**:超额 RankIC 提升 ≥ 10%
- **成本**:1 次额外 tokenizer 训练(约 2 小时)

### H4:lookback 长度
- **实验**:lookback ∈ {120, 240, 400}
- **预期**:240 最优(1 年周期信息量 vs 训练样本量平衡)

---

## ③ 实验执行(Experiment)

### 数据准备

```
个股日线:  /home/zxh/qlib_local_data/cn_data/features/{sh,sz,bj}*/  ✅ 已有
指数日线:  data/enhancement/index_883926.csv                          ⚠️ 待拉取(见 data-pipeline.md)
股池快照:  data/enhancement/universe_highbeta.csv                     ⚠️ 待拉取
```

**预处理产物**:`finetune/data/enhancement_daily/{fold0..3,full}/{train,val}.pkl`

每条样本结构:
```python
{
    'features': np.array(shape=[245, 12]),  # 240 lookback + 5 predict
    'time':     np.array(shape=[245, 5]),   # weekday/day/month + 2 占位
    'stock':    'SH600519',
}
# features[:, 0:6] = 个股 OHLCV(归一化)
# features[:, 6:12] = 指数 OHLCV(归一化,同期对齐)
```

### 训练配置(`config_daily.py`)

```python
class Config:
    # 数据源
    qlib_data_path_day = "/home/zxh/qlib_local_data/cn_data"
    index_csv = "data/enhancement/index_883926.csv"
    universe_csv = "data/enhancement/universe_highbeta.csv"

    # 窗口
    lookback_window = 240       # ~1 年交易日
    predict_window = 5          # 预测未来 5 天
    max_context = 512           # 240+5+1=246 ✅

    # 特征(关键:12 维)
    feature_list = ['open','high','low','close','vol','amt']                    # 个股
    index_feature_list = ['idx_open','idx_high','idx_low','idx_close','idx_vol','idx_amt']
    d_in = 12  # ← KronosTokenizer 需重新适配(见 3.3 注意点)

    # CV
    cv_folds = 4
    dataset_begin_time = "2020-03-02"
    dataset_end_time = "2026-05-15"

    # 训练
    batch_size = 32
    epochs = 8
    n_train_iter = 100000       # 日频样本少,多采样
    predictor_lr = 3e-5
```

> ⚠️ **3.3 注意点:d_in 从 6 改 12**
> 原始 KronosTokenizer 的 `d_in=6`(6 维 OHLCV)。本方案 d_in=12,tokenizer 第一层线性映射需要改。两种处理:
> - **方案 a(推荐)**:把个股和指数各自过独立的 6→d_model 映射,再相加(共享 tokenizer,不需重训)
> - **方案 b**:重新初始化 d_in=12 的 tokenizer,需 H3 两阶段微调
>
> Stage 1 先用方案 a 快速验证。

### 命令

```bash
cd /home/zxh/projects/Kronos/finetune

# 预处理(预计 10 分钟,日线数据小)
KRONOS_CONFIG=config_daily bash enhancement/run_daily.sh preprocess

# 冒烟测试(1 分钟,验证管线)
KRONOS_CONFIG=config_daily KRONOS_FOLD=0 python smoke_test.py

# 4 折训练(每折约 30 分钟,4 卡)
KRONOS_CONFIG=config_daily bash enhancement/run_daily.sh train-all

# 全量模型
KRONOS_CONFIG=config_daily bash enhancement/run_daily.sh train-final

# 回测 + 评估(输出超额 RankIC)
KRONOS_CONFIG=config_daily bash enhancement/run_daily.sh backtest
```

---

## ④ 度量评估(Measure)

每折训练后自动产出:

```
outputs/enhancement_daily/fold{N}/
├── finetune_predictor/
│   ├── checkpoints/best_model/
│   ├── summary.json          # Val Loss, 训练曲线
│   └── metrics.json          # ← 新增:超额 RankIC, IC 衰减
└── backtest/
    ├── daily_returns.csv     # 每日超额收益
    ├── positions.csv         # 每日持仓
    └── summary.json          # 累计收益/夏普/胜率/最大回撤
```

**门禁检查脚本** `enhancement/check_gate.py`:
```python
# 读取 fold2/fold3 的 metrics.json
# 判定: 超额 RankIC > 0.03 且两折都满足
# 输出: GATE_PASS / GATE_FAIL + 建议下一步
```

---

## ⑤ 反馈归因(Feedback)

无论通过/失败,都要做归因,决定下一轮假设:

### 如果超额 RankIC 达标(> 0.03)
- ✅ H1 证实,进入 Stage 2 分钟 Loop
- 但仍要检查:**IC 是否单调衰减**?**特定市场阶段是否失效**?

### 如果超额 RankIC 未达标
按归因决策树:

```
RankIC 接近 0?
├── 是 → 模型未学到横截面信息
│   ├── 检查:训练集 RankIC(是否过拟合)
│   ├── 假设:H3(tokenizer)/H4(lookback)
│   └── 若都无效 → Kronos 可能不适用此任务,考虑改架构
└── 否 → 学到反向信号
    ├── 检查:信号计算口径(是否又踩 v1/v2 bug)
    └── 检查:指数对齐错误(日期不匹配)

RankIC 高但不赚钱?
├── 检查:top-K 选股集中度(试 K=30)
├── 检查:交易成本假设(0.1% 是否过低)
└── 检查:IC 衰减速度(信号过期)
```

---

## ⑥ 迭代优化(Iterate)

### 单轮迭代流程(小步快跑)

```
假设 H_x → 改 1 个变量 → 训练 → 度量 → 归因 → 决策(通过/迭代/放弃)
                ↑                                    │
                └────────────────────────────────────┘
```

### 采样参数网格(无需重训,推理时调)

当模型训完后,在验证集扫这个网格找最优推理参数:

| 参数 | 网格 | 说明 |
|---|---|---|
| `T`(温度)| {0.1, 0.5, 1.0, 2.0} | 小 T 更确定 |
| `top_p` | {0.8, 0.9, 0.95} | 核采样阈值 |
| `sample_count` | {1, 5, 20} | 1 单点 / 20 平滑 |
| `K`(选股数)| {5, 10, 30, 50} | 集中度 |

### 重训触发条件

- 超额 RankIC 连续 2 折下降 > 30%
- 数据分布漂移(KS 检验 p < 0.05)
- 新增 1 个月以上新数据

---

## 7. 时间预算(预估)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 数据拉取 | 30 分钟 | iFinD 配额允许的话 |
| 日频预处理 | 10 分钟 | 日线数据小 |
| H1 基线训练 | 2 小时 | 4 折 × 30 分钟 |
| H1 对比训练 | 2 小时 | 12 维 vs 6 维 |
| 回测 + 归因 | 30 分钟 | |
| H3 tokenizer 微调 | 2 小时 | 若 H1 通过 |
| **Stage 1 总计** | **~7 小时** | 可一天内完成首轮 |

---

## 8. Stage 1 完成定义(Definition of Done)

满足以下**任一**条件即 Stage 1 完成:

- ✅ **成功路径**:超额 RankIC > 0.03 持续 2 折 → 进入 Stage 2
- ❌ **失败路径**:4 个假设全部证伪 → 归档结论"Kronos 不适用此任务",不再上 Stage 2
- 🟡 **部分成功**:RankIC 在 0~0.03 之间 → 记录基线,可选择性尝试 Stage 2(数据问题)

**不允许**:为了进入 Stage 2 而降低门禁标准。这是 Loop Engineer 的纪律。
