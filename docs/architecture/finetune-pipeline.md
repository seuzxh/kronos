# 本项目微调流水线设计

> 理解本项目如何组织一次 Kronos 微调实验。前置阅读:[kronos-overview.md](kronos-overview.md)
> 代码:[`/finetune/`](../../finetune/) 目录

---

## TL;DR

本项目基于原仓库的 `finetune_csv/`(单股票 CSV)扩展出 **`finetune/`(多股票 qlib 数据)** 路径,核心改动:
- 数据源从 CSV 改为 **qlib bin 格式**(支持上千只股票)
- 切分从固定 3 段改为 **k 折扩张式时间序列 CV**
- 配置/折数通过环境变量切换(`KRONOS_CONFIG` / `KRONOS_FOLD`)

流水线分 5 步:**预处理 → 验证 → 训练 → 推理 → 回测**。

---

## 1. 两条微调路径对比

原仓库和本项目的微调代码分两个目录:

| 维度 | `finetune_csv/`(原仓库) | `finetune/`(本项目扩展) |
|---|---|---|
| 数据格式 | 单只股票 CSV | qlib bin(多股票) |
| 适合场景 | 单股实验、教学 | 股池策略、大规模训练 |
| 数据切分 | 固定 3 段(train/val/test) | k 折扩张式时间序列 CV |
| 分布式 | 支持 DDP | 支持 DDP + sglang 共存 |
| 配置切换 | 改代码 | 环境变量 `KRONOS_CONFIG` |

**本项目所有微调方案都用 `finetune/` 路径**。

---

## 2. 流水线总览(5 步)

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. 预处理     │ → │ 2. 数据验证   │ → │ 3. 训练       │ → │ 4. 推理预测   │ → │ 5. 回测      │
│ preprocess   │   │ check_data   │   │ train_*      │   │ (predict)    │   │ backtest     │
│ 1min→5min    │   │ OHLC一致性   │   │ DDP 4卡      │   │              │   │ 日级调仓     │
│ CV切分        │   │              │   │              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
   preprocess_        check_data         train_           内嵌在           backtest_
   5min.py            .py                predictor.py     backtest         5min.py
   qlib_data_                            train_tokenizer  _5min.py
   preprocess.py                         .py
```

### 一键全流程

```bash
bash finetune/run_cv.sh all        # 预处理 → 4折训练 → 最终模型
```

### 分步执行

```bash
bash finetune/run_cv.sh preprocess        # 仅预处理
bash finetune/run_cv.sh check 0           # 验证 fold0 数据
bash finetune/run_cv.sh train 0           # 训练 fold0
bash finetune/run_cv.sh train-all         # 顺序训练 4 折
bash finetune/run_cv.sh train-final       # 训练最终全量模型
```

---

## 3. 各步骤详解

### 3.1 预处理(`preprocess_5min.py`)

输入:qlib 1min bin 数据 + 股池快照 CSV。
输出:`finetune/data/<方案>/{fold0..3,full}/{train,val}_5min.pkl`

关键逻辑:
1. **读 1min bin**:直接 `np.frombuffer` 读 bin 文件(绕过 qlib API,快 30 倍)
2. **1min → 5min 聚合**:按位置每 5 根聚 1 根(OHLC = first/max/min/last,vol = sum)
3. **CV 切分**:按交易日轴扩张式切 4 折,每折 20 日验证段
4. **序列化**:每股票一个连续 5min 序列(保留跨日),pickle 保存

> ⚠️ 关键坑(踩过):
> - qlib 1min 查询单日 end 必须跨到次日,否则返回空
> - 不能用 `resample('5min')`(产生 50 根),必须**按位置分组**(每 5 根聚 1 根,得 48 根)
> - `vwap` 在 qlib bin 里是**不复权**的,与后复权 close 口径冲突,本项目用 `amt = close * volume` 替代

### 3.2 数据验证(`check_data.py`)

```bash
python finetune/check_data.py --fold 0
```

检查项:
- 每天是否正好 48 根 5min(异常短日会标注)
- OHLC 一致性(high ≥ open/close/low,low ≤ open/close/high)
- 时间范围是否符合该折的切分边界
- 可用窗口数(序列长 - 289 + 1)

### 3.3 训练(`train_predictor.py` + `train_tokenizer.py`)

默认:**只训练 predictor,tokenizer 用预训练的**(可在 `run_cv.sh` 改成两阶段)。

```bash
KRONOS_CONFIG=config_highbeta KRONOS_FOLD=0 \
HF_HUB_OFFLINE=1 \
torchrun --standalone --nproc_per_node=4 train_predictor.py
```

- **DDP**:4 卡数据并行,有效 batch = 32×4 = 128
- **环境变量切换**:`KRONOS_CONFIG` 选 config 类,`KRONOS_FOLD` 选数据折
- **保存路径**:`outputs/<方案>/fold{N}/finetune_predictor/checkpoints/best_model/`
- **日志**:每 epoch 打印 train/val loss,收敛后写 `summary.json`

### 3.4 推理预测

不单独脚本,内嵌在回测脚本里。每次预测流程:
1. 加载某折 best_model
2. 对验证段每只股票,用前 240 根 5min 预测后 48 根
3. `sample_count=20` 次采样取均值
4. 反归一化 → 真实预测 K 线

### 3.5 回测(`backtest_5min.py`)

```bash
python finetune/backtest_5min.py --fold 0
python finetune/summarize_backtest.py   # 汇总 4 折
```

- **调仓**:日级(前日收盘预测,当日开盘买收盘卖)
- **选股**:预测涨幅 top-10 等权
- **信号**:反归一化到真实价格的预测涨幅
- **手续费**:0.1% 单边
- **输出**:累计收益、夏普、最大回撤、胜率

---

## 4. 配置体系

### 4.1 config 类继承

每个微调方案写一个 `config_<方案名>.py`,继承基础结构:

```python
# finetune/config_highbeta.py
class Config:
    qlib_data_path_1min = "/home/zxh/cn_data_1min"   # 数据源
    universe_csv = "..."                              # 股池
    lookback_window = 240                             # 窗口参数
    predict_window = 48
    cv_folds = 4                                      # CV 切分
    # ... 超参
```

通过环境变量切换:
```bash
KRONOS_CONFIG=config_highbeta python train_predictor.py
```

### 4.2 CV 切分设计(扩张式时间序列)

```
交易日轴  第1日 ────────────────────────────── 第609日

折叠0: [训练469日 ████][验证20日▓▓]              val=第470-489日
折叠1: [训练489日 ████████][验证20日▓▓]          val=第490-509日
折叠2: [训练509日 ████████████][验证20日▓▓]      val=第510-529日
折叠3: [训练529日 ████████████████][验证20日▓▓]  val=第590-609日  ← 最近
最终:  [训练609日 全量 ████████████████████████]  → 上线
```

**扩张式优点**:训练段递增,验证段连续滚动,**无未来数据泄露**(训练终点 < 验证起点)。

> 详细切分边界见各方案的 [design.md](../finetune/2026-07-highbeta-5min/design.md)。

---

## 5. DDP + sglang 共存配置(本项目踩坑)

服务器上 sglang 常驻服务占满 GPU 显存,直接跑 DDP 会 CUDA context 冲突。解决方案:

```bash
# finetune/run_cv.sh 顶部
export NCCL_P2P_DISABLE=1       # 禁用 P2P
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand
export NCCL_SHM_DISABLE=1       # 禁用共享内存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

这套配置下 4 卡 DDP + sglang 共占可稳定运行(训练每卡约 10GB,sglang 每卡约 84GB)。

---

## 6. 关键文件清单

| 文件 | 作用 |
|---|---|
| [`finetune/config.py`](../../finetune/config.py) | 基础 config(原仓库) |
| [`finetune/config_highbeta.py`](../../finetune/config_highbeta.py) | 高贝塔 5min 方案 config |
| [`finetune/universe.py`](../../finetune/universe.py) | 股池加载 |
| [`finetune/preprocess_5min.py`](../../finetune/preprocess_5min.py) | 1min→5min 聚合 + CV 切分 |
| [`finetune/check_data.py`](../../finetune/check_data.py) | 数据验证 |
| [`finetune/dataset.py`](../../finetune/dataset.py) | Dataset 类(归一化、窗口采样) |
| [`finetune/train_predictor.py`](../../finetune/train_predictor.py) | 训练 predictor |
| [`finetune/train_tokenizer.py`](../../finetune/train_tokenizer.py) | 训练 tokenizer(可选) |
| [`finetune/backtest_5min.py`](../../finetune/backtest_5min.py) | 5min 回测 |
| [`finetune/summarize_backtest.py`](../../finetune/summarize_backtest.py) | 4 折回测汇总 |
| [`finetune/run_cv.sh`](../../finetune/run_cv.sh) | CV 训练流水线编排 |
| [`finetune/smoke_test.py`](../../finetune/smoke_test.py) | 端到端冒烟测试 |

---

## 7. 如何新增一个微调方案

1. **写 config**:复制 `config_highbeta.py` 改名 `config_<新方案>.py`,调整数据源/股池/窗口/超参
2. **建文档目录**:`docs/finetune/YYYY-MM-<方案名>/`,复制 [`_TEMPLATE.md`](../finetune/_TEMPLATE.md) 填写
3. **预处理**:确认数据源 → `bash finetune/run_cv.sh preprocess`(可能需小改 preprocess 脚本)
4. **训练**:`KRONOS_CONFIG=config_<新方案> bash finetune/run_cv.sh train-all`
5. **回测**:写/改 backtest 脚本 → 跑 4 折 → 汇总
6. **归档**:把训练/回测报告填进方案文档目录,在 [`finetune/README.md`](../finetune/README.md) 索引表登记

详见 [_TEMPLATE.md](../finetune/_TEMPLATE.md)。
