# 高贝塔指增 Kronos 微调方案 - 方案总览

> **状态**:🟡 方案设计完成,待数据拉取与执行
> **日期**:2026-07-20
> **分支**:待创建 `feat/highbeta-enhancement`(从 master 切出)
> **前置方案**:[2026-07-highbeta-5min](../2026-07-highbeta-5min/README.md)(❌ 失败,本方案是其改进版)

---

## TL;DR

- **做什么**:以高贝塔指数(883926.TI)为基准,用 Kronos 做指数增强。微调目标从"预测 K 线形态"改为"预测个股相对指数的**超额收益**",直击上次实验暴露的核心缺陷。
- **两阶段推进**:
  - **Stage 1 日频**(主战场):数据 6 年充足,先验证指增思路是否 work
  - **Stage 2 分钟**:日频跑通且赚钱后,再上分钟级,**门禁严格**(上次 5min 失败主因是数据仅 4 个月)
- **关键创新**:引入**指数基准特征**和**超额收益信号**,把 Kronos 从"形态预测器"重塑为"相对强弱预测器"。
- **预期结论**:无法保证赚钱(金融预测本质概率问题),但通过 Loop 工程化能**系统性地找到能力边界**,而非像上次那样事后才发现"Val Loss 低 ≠ 能赚钱"。

> 💡 本方案是对上次 5min 失败的直接回应。所有设计决策都标注了"为什么与上次不同"。

---

## 1. 与上次 5min 方案的本质区别

| 维度 | 上次(2026-07-highbeta-5min)❌ | 本次(enhancement)|
|---|---|---|
| **预测目标** | 次日 48 根 5min K 线 token | **个股相对指数的超额收益** |
| **核心信号** | 预测涨幅 top-10 | **预测超额收益 top-K**(剥离指数 beta)|
| **频率** | 5min(数据仅 4 个月) | **日频优先**(6 年数据),分钟后置 |
| **基准** | 无 | **高贝塔指数 883926.TI** |
| **特征** | 纯个股 OHLCV | 个股 OHLCV + **指数同期 OHLCV 作为条件** |
| **评估指标** | Val Loss + 累计收益 | **RankIC(超额收益)** + 超额累计收益 + 夏普 |
| **失败模式** | Val Loss 低却不赚钱 | 用超额 RankIC 提前发现"学的是 alpha 还是 beta" |

**核心改进逻辑**:上次模型把所有股票都预测成相似的"均值回归形态",失去横截面区分度。本次直接优化"相对指数的强弱",让模型必须学到横截面 alpha 才能降 loss。

---

## 2. 数据资产清单(待拉取/确认)

| 数据 | 状态 | 来源 | 时间范围 |
|---|---|---|---|
| 高贝塔成分股快照(609 日) | ⚠️ 待拉取 | iFinD `data_pool` 接口 | 2024-01-02 ~ 2026-07-10 |
| 高贝塔指数日线(883926.TI) | ⚠️ 待拉取 | iFinD `date_sequence` | 同上 |
| 个股日线(后复权) | ✅ 已有 | `/home/zxh/qlib_local_data/cn_data` | 2020-03-02 ~ 2026-05-20 |
| 个股 5min/1min | ✅ 已有 | 同上 | 2026-01-05 ~ 2026-05-15 |
| 交易日历 | ✅ 已有 | qlib `day.txt` | — |

详见 [data-pipeline.md](data-pipeline.md)。

---

## 3. Loop 总体设计(六阶段闭环)

本方案不是一个"训练完看结果"的单次实验,而是**可迭代的闭环**。每个阶段都有明确的度量指标和迭代触发条件。

```
┌─────────────────────────────────────────────────────────────────────┐
│                     高贝塔指增 Kronos Loop                          │
│                                                                     │
│  ① 目标: 日频超额 RankIC > 0.03 (弱 alpha 起点)                    │
│     ↓                                                              │
│  ② 假设: 见 design.md 的 4 个可证伪假设                             │
│     ↓                                                              │
│  ③ 实验: 日频 Loop → (门禁通过)→ 分钟 Loop                         │
│     ↓                                                              │
│  ④ 度量: RankIC / 超额夏普 / 胜率 / IC 衰减                        │
│     ↓                                                              │
│  ⑤ 反馈: 归因(系统偏差/方差/漂移)→ 假设证伪/证实                   │
│     ↓                                                              │
│  ⑥ 迭代: 采样参数网格 → 特征调整 → 重训触发                        │
│                                                                     │
│  门禁(Stage 1 → Stage 2): 日频超额 RankIC > 0.03 持续 2 折        │
└─────────────────────────────────────────────────────────────────────┘
```

### 三个子 Loop

| Loop | 文档 | 目标 |
|---|---|---|
| 🔄 **日频 Loop**(Stage 1) | [loop-stage1-daily.md](loop-stage1-daily.md) | 验证指增思路,达到日频超额 RankIC > 0.03 |
| 🔄 **分钟 Loop**(Stage 2) | [loop-stage2-minute.md](loop-stage2-minute.md) | 日频跑通后,用分钟级增强信号(严格门禁)|
| 📥 **数据 Loop**(支撑) | [data-pipeline.md](data-pipeline.md) | iFinD 拉股池+指数,定期增量更新 |

---

## 4. 文档导航

- 📐 [design.md](design.md) — 指增思路与预测超额收益的架构设计(核心创新点)
- 🔄 [loop-stage1-daily.md](loop-stage1-daily.md) — 日频 Loop 详细设计(当前主战场)
- 🔄 [loop-stage2-minute.md](loop-stage2-minute.md) — 分钟 Loop 详细设计(Stage 2,含门禁)
- 📥 [data-pipeline.md](data-pipeline.md) — iFinD 数据拉取脚本设计(股池 + 指数行情)

---

## 5. 执行顺序(操作指引)

```bash
# Step 0: 拉数据(需 iFinD 账号在场,详见 data-pipeline.md)
bash finetune/enhancement/fetch_data.sh

# Step 1: 日频预处理 + 训练(详见 loop-stage1-daily.md)
bash finetune/enhancement/run_daily.sh preprocess
bash finetune/enhancement/run_daily.sh smoke
bash finetune/enhancement/run_daily.sh train-all

# Step 2: 日频回测 + 评估门禁
bash finetune/enhancement/run_daily.sh backtest
# 决策点:超额 RankIC > 0.03 持续 2 折? → 进入 Step 3,否则迭代日频 Loop

# Step 3: (门禁通过后)分钟 Loop
bash finetune/enhancement/run_minute.sh preprocess
bash finetune/enhancement/run_minute.sh train-all
```

---

## 6. 风险提示与免责

1. **金融预测本质是概率问题**:即使方案设计完美,也无法保证赚钱。RankIC > 0.03 只是"弱 alpha",实盘需考虑成本、滑点、容量。
2. **高贝塔股池信噪比低**:上次实验已证明该任务难度极大。本方案改进了预测目标,但不保证突破。
3. **数据依赖**:依赖 iFinD 账号配额,可能撞配额限制(见 data-pipeline.md 的配额管理)。
4. **不构成投资建议**:本方案仅供研究与学习,实盘需独立尽调。

---

## 7. 与已有方案的关系

- **继承**:`finetune/config_highbeta.py` / `preprocess_5min.py` / `dataset.py` / `backtest_5min.py` 的工程基础(DDP、qlib bin 直读、numpy 聚合等)
- **新增**:`finetune/enhancement/` 目录,含指数基准加载、超额收益计算、新 config
- **复盘**:认真对待 [上次 backtest-report.md](../2026-07-highbeta-5min/backtest-report.md) 的 9 条改进方向,本方案直接落地了其中第 1/2/3/5 条
