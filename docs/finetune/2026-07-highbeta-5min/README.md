# 高贝塔股池 5min Kronos 微调 - 方案总览

> **状态**:❌ 已回测,不具备实战价值
> **日期**:2026-07-16(训练) ~ 2026-07-18(回测)
> **分支**:`feat/highbeta-5min-finetune`(从 master `67b630e` 切出)

---

## TL;DR

- **做了什么**:微调 Kronos-base(102M),在高贝塔股池(883926.TI)上预测**次日全天 48 根 5min K 线走势**,4 折扩张式 CV + 最终全量模型。
- **核心结论**:❌ **不具备实战价值**。4 折合并 80 天累计收益 **-32.55%**,夏普 -3.79,胜率仅 40%。
- **关键发现**:**Val Loss 低(2.1964)≠ 能赚钱**。模型学会了"预测 5min K 线形态"(token 重建误差低),但**没学到横截面选股 alpha**(选出的 top-10 表现反而不如随机)。

> 💡 这是一次**有价值的失败**:完整验证了 Kronos 在 5min 高贝塔股池的能力边界,为后续改进提供清晰基线。

---

## 1. 方案配置

| 维度 | 值 |
|---|---|
| 股池 | 高贝塔(883926.TI),每日 100 只,日频换手 |
| 频率 | 5min(由 1min 聚合,48 根/天)|
| 数据时间范围 | 2024-01-02 ~ 2026-07-10(609 个交易日)|
| 股票数 | 历史并集 5114 只,有 1min 数据的 5038 只 |
| 窗口 | lookback=240(5 交易日) → predict=48(次日全天)|
| CV 切分 | 4 折扩张式时间序列 + 最终全量模型 |
| 模型 | Kronos-base(102M)|
| Tokenizer | **预训练**(未微调,疑点之一) |

### 关键超参

| 参数 | 值 |
|---|---|
| GPU | 4× H20(sglang 共占 84GB/卡,训练约 10GB/卡)|
| batch_size | 32(有效 batch 128,4 卡)|
| epochs | 8(OneCycleLR)|
| n_train_iter | 64000 样本/epoch |
| predictor_lr | 3e-5 |

---

## 2. 核心结果

### 训练结果

| 模型 | Val Loss | 验证段 | 训练耗时 |
|---|---|---|---|
| fold0 | **2.1887** ← 最佳单折 | 2026-03-16~04-13 | 38min |
| fold1 | 2.1966 | 2026-04-14~05-14 | 41min |
| fold2 | 2.2780 ⚠️ 异常偏高 | 2026-05-15~06-11 | 35min |
| fold3 | 2.2002 | 2026-06-12~07-10 | 44min |
| **full** | **2.1964** ← 推荐上线 | (复用 fold3 验证段) | 50min |

- 4 折平均 **2.2159**,zero-shot 基线 2.35 → **微调降幅 6.8%**
- 各折 epoch 7-8 收敛,Val Loss 单调下降,**无过拟合**

### 回测结果(v2,修复信号后)

| Fold | 验证段 | 累计收益 | 夏普 | 最大回撤 | 胜率 | 日均 |
|---|---|---|---|---|---|---|
| fold0 | 03-16~04-13 | **-9.39%** | -3.64 | -16.27% | 35% | -0.47% |
| fold1 | 04-14~05-14 | **-11.27%** | -4.82 | -11.49% | 45% | -0.58% |
| fold2 | 05-15~06-11 | **-6.40%** | -2.77 | -8.74% | 40% | -0.31% |
| fold3 | 06-12~07-10 | **-10.37%** | -3.92 | -13.76% | 40% | -0.52% |
| **合并** | **80 天** | **-32.55%** | **-3.79** | **-36.34%** | **40%** | **-0.47%** |

**4 折全部亏损,策略稳定亏损。** 胜率 40% < 50%,说明模型选股方向系统性有误。

---

## 3. 文档导航

完整档案:

- 📐 [design.md](design.md) — 方案设计(CV 切分边界、数据处理细节、关键技术决策)
- 🛠️ [devlog.md](devlog.md) — 开发日志(qlib/聚合/性能优化等踩坑全记录)
- 📈 [training-report.md](training-report.md) — 训练结果分析(各折收敛曲线、健康度评估)
- 💰 [backtest-report.md](backtest-report.md) — 回测结果分析(为什么亏、归因、改进方向)

---

## 4. 改进方向

详细分析见 [backtest-report.md](backtest-report.md),这里给概览:

### 🔴 可能需要根本性改变(成本高)
1. **改预测目标**:从"预测 5min K 线 token"改为"直接预测次日涨跌幅排名"(偏离 Kronos 架构)
2. **加入横截面信息**:当前每只股票独立预测,改为 ranking loss 或加市场特征

### 🟡 值得尝试(中成本)
3. **微调 tokenizer**:当前用预训练的,可能 5min 分布失真
4. **融合手工因子**:把 qlib_ifind_beta 的 18 个分钟因子作为额外特征
5. **更长持仓**:试 T+3/T+5(可能预测准的是趋势而非日内)
6. **不同 top-K**:top-10 太集中,试 top-30

### 🟢 快速验证(低成本)
7. **降低 sample_count**:当前 20 次取均值可能过度平滑,试 sample=1
8. **调推理温度 T**:当前 T=1.0,试 T=0.1 或 T=5
9. **多信号组合**:close+high+low 综合打分

> 📊 对比基准:本项目作者的 `qlib_ifind_beta`(LGBM + 18 手工因子)IC 0.067,超额 +177%,远优于当前 Kronos 方案。

---

## 5. 复现命令

```bash
cd /home/zxh/projects/Kronos/finetune

# 1. 预处理(约 27 分钟)
bash run_cv.sh preprocess

# 2. 数据验证
bash run_cv.sh check 0
bash run_cv.sh check full

# 3. 冒烟测试(约 1 分钟,验证训练管线)
KRONOS_FOLD=0 python smoke_test.py

# 4. 4 折 CV 训练(约 2.5 小时,4 卡)
bash run_cv.sh train-all

# 5. 最终全量模型
bash run_cv.sh train-final

# 6. 回测
python backtest_5min.py --fold 0
python summarize_backtest.py
```

---

## 6. 产物位置

```
finetune/data/highbeta_5min/{fold0..3,full}/{train,val}_5min.pkl    # 预处理产物
finetune/outputs/highbeta_5min/fold{0..3,full}/finetune_predictor/  # 模型产物
outputs/highbeta_5min/TRAINING_REPORT.md                            # 训练报告
outputs/highbeta_5min/BACKTEST_REPORT.md                            # 回测报告
outputs/highbeta_5min/train_fold*.log                               # 训练日志
outputs/highbeta_5min/backtest_v2_fold*.log                         # 回测日志
```

> ⚠️ `finetune/data/` 和 `finetune/outputs/` 被 .gitignore 排除(体积大,约 23GB),仅 `outputs/highbeta_5min/` 下的报告和日志入 git。

---

## 7. 客观评价

这次实验的价值**不在于做出赚钱策略**,而在于:

1. ✅ 完整验证了 Kronos 在 5min 高贝塔股池的能力边界
2. ✅ 证明了"Val Loss 低 ≠ 能赚钱"(形态学得会 ≠ 选股 alpha)
3. ✅ 为后续改进提供了清晰基线和方向
4. ✅ 对比基准:LGBM(IC 0.067,超额 +177%)远优于当前 Kronos 方案
