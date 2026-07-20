# <方案名> - 微调方案总览

> **状态**:🟡 待定 | ✅ 上线 | ❌ 失败
> **日期**:YYYY-MM-DD
> **作者**:<名字>

---

## TL;DR(三句话讲清楚)

- **做了什么**:<用一句话描述,如 "微调 Kronos-base 在 XX 股池 YY 频率上预测 ZZ">
- **核心结论**:<Val Loss 多少 / 回测收益多少 / 能不能上线>
- **关键发现**:<最重要的 1-2 条经验教训>

---

## 1. 方案配置

| 维度 | 值 |
|---|---|
| 股池 | <如 高贝塔 883926.TI,每日 100 只> |
| 频率 | <如 5min(由 1min 聚合)> |
| 数据时间范围 | <如 2024-01-02 ~ 2026-07-10> |
| 窗口 | lookback=<240> → predict=<48> |
| CV 切分 | <如 4 折扩张式 + 最终全量> |
| 模型 | Kronos-base(102M)|
| Tokenizer | <预训练 / 本方案微调> |

### 关键超参

| 参数 | 值 | 备注 |
|---|---|---|
| batch_size | | |
| epochs | | |
| lr | | |
| n_train_iter | | |
| GPU | | |

---

## 2. 核心结果

### 训练结果

| 模型 | Val Loss | 备注 |
|---|---|---|
| fold0 | | |
| fold1 | | |
| fold2 | | |
| fold3 | | |
| full | | |

### 回测结果

| Fold | 累计收益 | 夏普 | 最大回撤 | 胜率 |
|---|---|---|---|---|
| fold0 | | | | |
| ... | | | | |
| **合并** | | | | |

---

## 3. 文档导航

本方案完整档案:

- 📐 design.md — 方案设计(CV 切分、数据处理、超参选择理由)
- 🛠️ devlog.md — 开发日志(实施细节、踩坑记录)
- 📈 training-report.md — 训练结果分析(收敛曲线、健康度)
- 💰 backtest-report.md — 回测结果分析(收益归因、改进方向)

---

## 4. 改进方向

<列出本方案暴露的问题和后续改进方向,按优先级排序>

### 🔴 高优先级
1. ...

### 🟡 中优先级
2. ...

### 🟢 低优先级
3. ...

---

## 5. 复现命令

```bash
# 预处理
bash finetune/run_cv.sh preprocess

# 训练
KRONOS_CONFIG=config_<方案> bash finetune/run_cv.sh train-all
KRONOS_CONFIG=config_<方案> bash finetune/run_cv.sh train-final

# 回测
python finetune/backtest_5min.py --fold 0
python finetune/summarize_backtest.py
```

---

## 6. 产物位置

```
finetune/data/<方案>/{fold0..3,full}/{train,val}_5min.pkl    # 预处理产物(不入 git)
finetune/outputs/<方案>/fold{N}/finetune_predictor/           # 模型产物(不入 git)
outputs/<方案>/TRAINING_REPORT.md                             # 自动生成的训练报告
outputs/<方案>/BACKTEST_REPORT.md                             # 自动生成的回测报告
```

---

<!--
模板使用说明:
1. 复制本文件到新方案目录,改名为 README.md
2. 填充所有 <> 占位符
3. 删除不需要的章节,但保留 TL;DR / 配置 / 结果 / 复现命令
4. 详细内容拆到 design.md / devlog.md / training-report.md / backtest-report.md
5. 完成后在本目录上级的 finetune/README.md 索引表登记一行
-->
