# 高贝塔指增 Kronos 微调方案 - 方案总览

> **状态**:❌ Stage 1 门禁未通过(4 折平均超额 RankIC -0.006,目标 > 0.03)
> **日期**:2026-07-20
> **分支**:`feat/highbeta-5min-finetune`(代码与 5min 方案共用分支)
> **前置方案**:[2026-07-highbeta-5min](../2026-07-highbeta-5min/README.md)(❌ 失败,本方案是其改进版)

---

## TL;DR

- **做了什么**:以高贝塔指数(883926.TI)为基准,微调 Kronos-base 预测日频超额收益,4 折 CV + full 模型。
- **核心结论**:❌ **Stage 1 门禁未通过**。4 折平均超额 RankIC = **-0.006**(目标 > 0.03),4 个单折全部 ≤ 0,只有 full 模型刚过线(0.034)。
- **关键发现**:
  1. 单纯增加数据量(4 个月 → 2.5 年)**不能**解决"预测形态 ≠ 选股 alpha"的根本问题
  2. 累计收益 +16% 但 RankIC 负 = 假 alpha(市场 beta / 运气)
  3. **建立了正确的评估方法**(超额 RankIC),这是比"做出策略"更有价值的产出
- **下一步**:验证 H4(微调 tokenizer)+ predict_window=1,若仍无改善则归档

> 💡 本方案是对上次 5min 失败的直接回应。虽然门禁未过,但**建立了正确的评估框架**,暴露了真相。

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

## 2. 数据资产清单

| 数据 | 状态 | 来源 | 时间范围 |
|---|---|---|---|
| 高贝塔成分股快照(609 日) | 🟢 脚本就绪,实测通过 | iFinD `p03473` 接口 | 2024-01-02 ~ 2026-07-17 |
| 高贝塔指数日线(883926.TI) | 🟢 脚本就绪,实测通过 | iFinD `fetch_history_quotation` | 同上 |
| 个股日线(后复权) | ✅ 已有 | `/home/zxh/qlib_local_data/cn_data` | 2020-03-02 ~ 2026-05-20 |
| 个股 5min/1min | ✅ 已有 | 同上 | 2026-01-05 ~ 2026-05-15 |
| 交易日历 | ✅ 已有 | qlib `day.txt` | — |

**实测验证(2026-07-20)**:
- p03473 接口拉成分股:3 天 × 100 股 = 300 行 ✅,字段含义已确认(f001=日期/f002=代码/f003=名称)
- fetch_history_quotation 拉指数日线:6 行 ✅,OHLC 一致性通过
- 成分股 code_qlib 与 qlib 数据交叉验证:261/263(99.2%)有日线目录 ✅

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
# Step 0: 拉数据(脚本已就绪,端到端实测通过)
bash finetune/enhancement/fetch_data.sh                    # 拉全部(~5 分钟)
# 或分步:
bash finetune/enhancement/fetch_data.sh index              # 仅指数日线
bash finetune/enhancement/fetch_data.sh universe           # 仅成分股快照
bash finetune/enhancement/fetch_data.sh universe --resume  # 断点续拉

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
