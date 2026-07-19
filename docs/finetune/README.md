# Kronos 微调方案记录

> 本目录记录每次 Kronos 微调方案的完整档案:**设计 → 训练 → 回测 → 结论**全链路。
> 目标:让任何人在任何时候都能复现、复盘、对比历次方案。

---

## 📋 方案索引

| 方案 | 日期 | 股池 | 频率 | 窗口 | Val Loss | 回测收益 | 状态 |
|---|---|---|---|---|---|---|---|
| [高贝塔 5min](2026-07-highbeta-5min/README.md) | 2026-07 | 高贝塔(883926.TI) | 5min | 240→48 | **2.1964** | **-32.55%** | ❌ 无实战价值 |
| [高贝塔指增](2026-07-highbeta-enhancement/README.md) | 2026-07 | 高贝塔(883926.TI) | 日频 | 240→5 | 2.0184 | RankIC -0.006 | ❌ 门禁未通过 |

> **状态图例**:✅ 上线 / 🟡 待改进 / ❌ 失败 / 🧪 实验中

### 方案间关系

- **2026-07-highbeta-5min** ❌ → 首次尝试,失败(Val Loss 低却不赚钱)
- **2026-07-highbeta-enhancement** ❌ → 改进版,建立了正确的超额 RankIC 评估,但仍未通过门禁(4 折平均 -0.006)。证明"增加数据量 + 改预测目标"不足以解决 Kronos 选股 alpha 问题

### 核心经验沉淀(跨方案)

1. **Val Loss 低 ≠ 能赚钱**(高贝塔 5min 首次验证):token 重建误差低只说明"学会预测形态",不等于横截面选股 alpha
2. **5min 高贝塔股池信噪比极低**:每日换手 84%、分钟级噪声大,Kronos 直接从 K 线预测难度极大
3. **预处理是性能瓶颈**:直接读 qlib bin + numpy 聚合比 qlib API 快 30 倍

---

## 📁 方案目录结构(约定)

每次方案建一个子目录,命名 `YYYY-MM-<简称>`:

```
docs/finetune/
├── README.md                         # 本索引
├── _TEMPLATE.md                      # 新方案模板
└── 2026-07-highbeta-5min/            # 方案子目录
    ├── README.md                     # 方案总览(TL;DR + 配置 + 结论)
    ├── design.md                     # 方案设计(CV切分/数据处理/超参选择)
    ├── devlog.md                     # 开发日志(技术细节/踩坑)
    ├── training-report.md            # 训练结果分析(各折曲线/健康度)
    └── backtest-report.md            # 回测结果分析(收益/归因/改进方向)
```

### 各文档定位

| 文档 | 回答什么问题 | 读者 |
|---|---|---|
| `README.md` | 这方案做了啥、结论是啥 | 所有人(决策者先看)|
| `design.md` | 为什么这么设计、超参怎么选 | 想复现/改进的人 |
| `devlog.md` | 实施细节、踩了什么坑 | 维护者、code review |
| `training-report.md` | 训练效果如何、模型健康吗 | 模型评估者 |
| `backtest-report.md` | 赚不赚钱、为什么 | 策略评估者 |

---

## 🆕 如何新增一个方案

### 步骤

1. **建目录**:`mkdir docs/finetune/$(date +%Y-%m)-<方案名>`
2. **复制模板**:`cp docs/finetune/_TEMPLATE.md docs/finetune/<方案目录>/README.md`
3. **写 config**:`cp finetune/config_highbeta.py finetune/config_<方案名>.py` 并修改
4. **跑实验**:预处理 → 训练 → 回测(参考 [finetune-pipeline.md](../architecture/finetune-pipeline.md))
5. **填文档**:根据实验结果填 README,另写 design / devlog / training-report / backtest-report
6. **登记索引**:在本文件上方"方案索引"表加一行

### 命名约定

- 方案名用小写连字符: `highbeta-5min` / `csi300-daily` / `crypto-1h`
- 不用空格、中文、下划线
- 模型变体(如换 tokenizer)在同一方案下用 `-v2` 后缀,不开新方案

### 文档写作约定

- **TL;DR / 结论先行**:每篇开头用一段话总结核心结论
- **数据可复现**:关键超参、数据路径、CV 切分、命令行都要写清楚
- **诚实记录**:失败也要记,说明为什么不 work
- **不堆砌日志**:复制粘贴训练日志要有提炼,不要无脑堆叠
