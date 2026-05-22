# Kronos 系统架构设计文档

## 0. 模型主要作用与架构分类

### 0.1 主要作用

Kronos 是一个**基于 Token 化的自回归时间序列预测模型**，核心作用是将金融 K 线数据转化为离散 token 序列，然后用 Transformer 进行自回归预测。与传统时间序列模型不同，Kronos 借鉴了大语言模型（LLM）的架构思想：

1. **Tokenizer（分词器）**：将连续的 OHLCV+A 数值序列编码为离散 token ID
2. **Kronos Model（预测模型）**：基于 token 序列进行自回归生成，预测未来 token
3. **解码**：将预测的 token ID 解码回 OHLCV+A 数值

### 0.2 架构分类

Kronos 系统由三个核心组件构成：

| 组件 | 类名 | 作用 | 源码位置 |
|------|------|------|---------|
| **分词器** | `KronosTokenizer` | 将连续数值编码为离散 token，或将 token 解码回数值 | `model/kronos.py` L13-177 |
| **预测模型** | `Kronos` | 基于历史 token 自回归预测未来 token | `model/kronos.py` L180-296 |
| **预测器** | `KronosPredictor` | 封装完整预测流程（归一化→编码→推理→解码→反归一化） | `model/kronos.py` L482-559 |

#### 分词器（KronosTokenizer）内部结构

| 子模块 | 类名 | 作用 |
|--------|------|------|
| Encoder | `TransformerBlock` × N | 将输入特征编码为隐表示 |
| 量化器 | `BSQuantizer` → `BinarySphericalQuantizer` | 将连续隐表示量化为离散 token（二值球面量化） |
| Decoder | `TransformerBlock` × N | 将量化表示解码回特征空间 |

分词器采用**分层量化**策略，将每个时间步的 6 维特征量化为两个 token：
- **s1 token（粗粒度）**：`s1_bits` 位，捕捉主要趋势和价格水平
- **s2 token（细粒度）**：`s2_bits` 位，捕捉细节波动

| 分词器版本 | s1_bits | s2_bits | s1 词表大小 | s2 词表大小 | 总 token 空间 |
|-----------|---------|---------|------------|------------|--------------|
| Kronos-Tokenizer-base | 10 | 10 | 1,024 | 1,024 | 1,048,576 |
| Kronos-Tokenizer-2k | 11 | 11 | 2,048 | 2,048 | 4,194,304 |

#### 预测模型（Kronos）内部结构

| 子模块 | 类名 | 作用 |
|--------|------|------|
| 层次嵌入 | `HierarchicalEmbedding` | 将 s1/s2 token ID 映射为向量，融合为统一表示 |
| 时间嵌入 | `TemporalEmbedding` | 编码时间特征（周期性、趋势） |
| Transformer | `TransformerBlock` × N | 自回归建模 token 序列依赖关系 |
| 依赖感知层 | `DependencyAwareLayer` | s2 解码时交叉关注 s1，保证粗细粒度一致性 |
| 双头输出 | `DualHead` | 分别输出 s1 logits 和 s2 logits |

推理流程：
1. 历史数据 → KronosTokenizer.encode() → s1_ids + s2_ids
2. Kronos.forward(s1_ids, s2_ids, stamp) → s1_logits, s2_logits
3. 采样 s1_id → DependencyAwareLayer → s2_logits → 采样 s2_id
4. 重复 2-3 直到生成 pred_len 个 token
5. KronosTokenizer.decode(s1_ids + s2_ids) → 预测数值

#### 预测器（KronosPredictor）封装流程

```
原始 DataFrame
  → 归一化（clip + z-score）
  → KronosTokenizer.encode()
  → 自回归推理（auto_regressive_inference）
  → KronosTokenizer.decode()
  → 反归一化
  → 预测 DataFrame
```

---

## 1. 功能来源标注表

| 模块/功能 | 来源 | 源码位置 | 说明 |
|-----------|------|----------|------|
| Kronos 核心模型 | [自带] | `model/kronos.py`, `model/module.py` | 基于 Transformer 的时间序列预测模型，使用层级嵌入 + 依赖感知层 + 双头输出（s1/s2） |
| KronosTokenizer | [自带] | `model/kronos.py` | 混合量化分词器，使用 BSQuantizer（Binary Spherical Quantization）将连续数据编码为离散 token |
| KronosPredictor.predict() | [自带] | `model/kronos.py` | 单序列预测接口，自动完成归一化、分词、自回归推理、反归一化全流程 |
| KronosPredictor.predict_batch() | [自带] | `model/kronos.py` | 批量并行预测接口，要求所有序列具有相同的历史长度和预测长度 |
| 自回归推理引擎 | [自带] | `model/kronos.py` (`auto_regressive_inference`) | 逐步生成 s1/s2 token，支持滑动窗口上下文管理 |
| Top-k/Top-p 采样 | [自带] | `model/kronos.py` (`top_k_top_p_filtering`, `sample_from_logits`) | 控制生成多样性的采样策略，支持温度缩放 |
| 数据归一化/反归一化 | [自带] | `model/kronos.py` (`KronosPredictor.predict`) | z-score 归一化 + clip 裁剪，预测后自动反归一化 |
| Web UI Flask 应用 | [自带] | `webui/app.py` | 基于 Flask 的 Web 界面，端口 7070，提供可视化预测操作 |
| 微调脚本 - Qlib | [自带] | `finetune/` | 基于 Qlib 数据的模型微调脚本 |
| 微调脚本 - CSV | [自带] | `finetune_csv/` | 基于 CSV 数据的模型微调脚本 |
| 预测示例 | [自带] | `examples/` | 包含单序列、批量、A股、AkShare 等多种预测示例 |
| 统一配置管理 | [新增] | `config.py` | 基于 dataclass + dotenv 的配置管理，支持环境变量覆盖，包含参数校验和设备自动检测 |
| LLM 分析器 | [新增] | `llm_analyzer.py` | 基于 litellm 的 LLM 分析模块，将预测结果发送至 LLM 生成趋势/支撑阻力/建议/风险分析 |
| 数据加载器 | [新增] | `data_loader.py` | CSV 数据加载与预处理，支持自动列名映射、缺失列填充、涨跌停后处理 |
| FastAPI 推理服务 | [新增] | `api.py` | 基于 FastAPI 的 REST API 服务，提供健康检查、模型状态、预测接口 |
| .env 配置文件 | [新增] | `.env` | 存储敏感配置（API Key 等），已加入 .gitignore |
| .env.example 配置示例 | [新增] | `.env.example` | 配置模板，包含所有可配置项及注释说明 |

---

## 2. 输入数据要求

### 2.1 列要求

| 类别 | 列名 | 是否必须 | 缺失处理 | 说明 |
|------|------|----------|----------|------|
| 价格列 | `open` | ✅ 必须 | 缺失抛出 ValueError | 开盘价 |
| 价格列 | `high` | ✅ 必须 | 缺失抛出 ValueError | 最高价 |
| 价格列 | `low` | ✅ 必须 | 缺失抛出 ValueError | 最低价 |
| 价格列 | `close` | ✅ 必须 | 缺失抛出 ValueError | 收盘价 |
| 交易列 | `volume` | ✅ 必须 | 缺失抛出 ValueError | 成交量 |
| 交易列 | `amount` | ❌ 可选 | volume 存在时估算为 `volume × 均价`，否则填充 0.0 | 成交额 |
| 时间列 | `timestamps` 或 `date` | ❌ 可选 | 需为 datetime 类型 | 时间戳，用于生成时间嵌入 |

### 2.2 模型内部维度

无论输入提供多少列，模型内部始终处理 **6 维**数据：

```
[open, high, low, close, volume, amount]
```

- `volume` 缺失 → 抛出 ValueError（必须提供）
- 仅 `amount` 缺失而 `volume` 存在 → 估算为 `volume × (open+high+low+close)/4`

### 2.3 数据量约束

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lookback` | 400 | 输入历史 K 线根数（small/base 模型） |
| `pred_len` | 20-120 | 预测未来 K 线根数 |
| `max_context` | 512（small/base）/ 2048（mini） | 模型上下文窗口长度 |
| `lookback + pred_len` | ≤ `max_context` | 历史长度加预测长度不能超过上下文窗口 |

### 2.5 批量预测限制

Kronos 代码中没有硬编码的批量大小限制，实际限制由 GPU 显存决定：

| 配置 | 估算最大股票数 | 说明 |
|------|--------------|------|
| `sample_count=1, lookback=400, pred_len=120` | ~50-100 | H20 80GB VRAM + Kronos-small |
| `sample_count=5, lookback=400, pred_len=120` | ~10-20 | 5 倍显存消耗 |
| `sample_count=1, lookback=400, pred_len=120` + Kronos-base | ~30-50 | 模型权重 ~8GB |

- 批量张量形状：`(B × sample_count, seq_len, 6)`，其中 B = 股票数量
- 每步自回归推理需对整个 batch 做一次前向传播
- 超出显存时 PyTorch 抛出 CUDA OOM 错误
- API 层通过 `MAX_BATCH_SIZE`（默认 50）限制单次批量预测数量

### 2.4 数据来源

| 方式 | 状态 | 说明 |
|------|------|------|
| CSV 文件 | ✅ 可用 | 本地 K 线数据，通过 `data_loader.load_csv()` 加载，支持自动列名映射 |
| qlib 本地目录 | ✅ 可用 | 通过 `data_loader.load_qlib()` 从 qlib 本地数据目录读取 OHLCV+A |

---

## 3. 数据获取方式

### 3.1 CSV 文件加载

通过 [data_loader.py](file:///home/zxh/quant_projects/kronos/data_loader.py) 的 `load_csv()` 函数加载：

```python
from data_loader import load_csv

df = load_csv("data.csv")
```

#### 自动列名映射

`data_loader.py` 内置列名映射表，支持多种常见列名自动转换：

| 原始列名 | 映射目标 |
|----------|----------|
| `date`, `datetime`, `timestamp`, `time` | `timestamps` |
| `open` | `open` |
| `high` | `high` |
| `low` | `low` |
| `close`, `close_price` | `close` |
| `volume`, `vol` | `volume` |
| `amount`, `amt`, `turnover` | `amount` |

#### 加载流程

1. 读取 CSV 文件
2. 根据映射表重命名列（大小写不敏感）
3. 将 `timestamps` 列转换为 datetime 类型
4. 检查必须列（open/high/low/close/volume），缺失则抛出 ValueError
5. 填充可选列（amount → 0.0）

### 3.2 qlib 本地目录加载

通过 [data_loader.py](file:///home/zxh/quant_projects/kronos/data_loader.py) 的 `load_qlib()` 函数加载：

```python
from data_loader import load_qlib

df = load_qlib(
    symbol="SH600977",
    start_time="2024-01-01",
    end_time="2025-05-16",
    provider_uri="/data02/home/zxh/qlib_local_data/cn_data",
)
```

#### qlib 字段自动映射

| qlib 字段 | Kronos 字段 | 说明 |
|-----------|------------|------|
| `$open` | `open` | 开盘价 |
| `$high` | `high` | 最高价 |
| `$low` | `low` | 最低价 |
| `$close` | `close` | 收盘价 |
| `$volume` | `volume` | 成交量 |
| `$amount` | `amount` | 成交额 |
| DataFrame 索引 `datetime` | `timestamps` | 日期时间 |

依赖：`pip install pyqlib`

### 3.3 数据获取方式汇总

| 方式 | 状态 | 说明 |
|------|------|------|
| CSV 文件 | ✅ 可用 | 本地 K 线数据，通过 `data_loader.load_csv()` 加载，支持自动列名映射 |
| qlib 本地目录 | ✅ 可用 | 通过 `data_loader.load_qlib()` 从 qlib 本地数据目录读取 OHLCV+A |

---

## 4. 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户层 (User Layer)                        │
│                                                                     │
│   ┌──────────────────┐              ┌──────────────────┐            │
│   │   API Client     │              │     Web UI       │            │
│   │  (curl/SDK/...)  │              │  (Flask :7070)   │            │
│   │                  │              │   [自带]          │            │
│   └────────┬─────────┘              └────────┬─────────┘            │
│            │                                  │                      │
└────────────┼──────────────────────────────────┼──────────────────────┘
             │ HTTP                             │ HTTP
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         服务层 (Service Layer)                      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              FastAPI 推理服务 (api.py)                    │      │
│   │                    [新增]                                 │      │
│   │                                                          │      │
│   │  GET  /api/health          → 健康检查                    │      │
│   │  GET  /api/model-status    → 模型状态查询                │      │
│   │  POST /api/predict         → 上传 CSV 执行预测           │      │
│   │  POST /api/predict-qlib   → 从 qlib 读取数据并预测      │      │
│   └──────────────────────────┬───────────────────────────────┘      │
│                               │                                     │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        业务层 (Business Layer)                      │
│                                                                     │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│   │   config.py    │  │ data_loader.py │  │llm_analyzer.py │       │
│   │    [新增]      │  │    [新增]      │  │    [新增]      │       │
│   │                │  │                │  │                │       │
│   │ • KronosConfig │  │ • load_csv()   │  │ • LLMAnalyzer  │       │
│   │ • load_config()│  │ • load_qlib()  │  │ • analyze_     │       │
│   │ • get_device() │  │ • apply_price_ │  │   prediction() │       │
│   │ • validate()   │  │ • COLUMN_      │  │ • is_available │       │
│   │ • is_llm_      │  │   MAPPING      │  │                │       │
│   │   configured() │  │                │  │                │       │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘       │
│           │                   │                   │                 │
└───────────┼───────────────────┼───────────────────┼─────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         核心层 (Core Layer)                         │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              KronosPredictor  [自带]                      │      │
│   │                                                          │      │
│   │  • predict()         单序列预测                          │      │
│   │  • predict_batch()   批量并行预测                        │      │
│   │  • generate()        底层推理调用                        │      │
│   │  • 归一化/反归一化    z-score + clip                     │      │
│   └──────────┬───────────────────────┬───────────────────────┘      │
│              │                       │                              │
│   ┌──────────▼──────────┐  ┌────────▼─────────┐                    │
│   │  KronosTokenizer    │  │   Kronos Model    │                    │
│   │     [自带]          │  │     [自带]        │                    │
│   │                     │  │                    │                    │
│   │ • encode()          │  │ • forward()        │                    │
│   │ • decode()          │  │ • decode_s1()      │                    │
│   │ • indices_to_bits() │  │ • decode_s2()      │                    │
│   │ • BSQuantizer       │  │ • HierarchicalEmb  │                    │
│   └─────────────────────┘  │ • TemporalEmbed    │                    │
│                            │ • DependencyLayer  │                    │
│                            │ • DualHead         │                    │
│                            └────────────────────┘                    │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │          自回归推理引擎  [自带]                           │      │
│   │                                                          │      │
│   │  auto_regressive_inference()                             │      │
│   │  • 滑动窗口上下文管理 (max_context)                      │      │
│   │  • s1 → s2 两阶段解码                                   │      │
│   │  • Top-k/Top-p 采样过滤                                 │      │
│   │  • 多次采样取均值 (sample_count)                         │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         数据层 (Data Layer)                         │
│                                                                     │
│   ┌────────────────┐              ┌────────────────┐                │
│   │    .env        │              │   CSV Files    │                │
│   │   [新增]       │              │                │                │
│   │                │              │  K线数据文件    │                │
│   │ • 模型配置     │              │  (OHLCV + 时间)│                │
│   │ • 预测参数     │              │                │                │
│   │ • LLM API Key  │              │                │                │
│   │ • 服务端口     │              │                │                │
│   └────────────────┘              └────────────────┘                │
│                                                                     │
│   ┌────────────────────────────────────────────────┐                │
│   │   qlib 本地数据目录                             │                │
│   │   /data02/home/zxh/qlib_local_data/cn_data     │                │
│   │   • features/ (OHLCV+A .day.bin)               │                │
│   │   • calendars/ (交易日历)                       │                │
│   │   • instruments/ (股票池)                       │                │
│   └────────────────────────────────────────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 数据流图

### 5.1 预测主流程

```
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐
│ CSV 文件  │────▶│ data_loader      │────▶│  清洗后 DataFrame  │
│          │     │ .load_csv()      │     │  [open,high,low,  │
│          │     │ [新增]           │     │   close,volume,   │
│          │     │ • 列名映射       │     │   amount,         │
│          │     │ • 类型转换       │     │   timestamps]     │
│          │     │ • 缺失填充       │     │                   │
└──────────┘     └──────────────────┘     └─────────┬─────────┘
                                                     │
                                                     │ 截取 lookback 行
                                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    KronosPredictor.predict()  [自带]              │
│                                                                  │
│  1. 校验必须列 (open/high/low/close/volume)                      │
│  2. 填充可选列 (amount → 估算或 0.0)                              │
│  3. z-score 归一化: x_norm = (x - mean) / (std + 1e-5)          │
│  4. clip 裁剪: x_norm = clip(x_norm, -clip, clip)               │
│  5. 时间特征提取: [minute, hour, weekday, day, month]            │
│  6. 自回归推理:                                                  │
│     a. tokenizer.encode(x) → (s1_ids, s2_ids)                   │
│     b. for i in range(pred_len):                                 │
│          model.decode_s1() → s1_logits → sample → s1_id         │
│          model.decode_s2() → s2_logits → sample → s2_id         │
│          更新滑动窗口缓冲区                                      │
│     c. tokenizer.decode([s1_ids, s2_ids]) → 预测向量             │
│     d. 多次采样取均值 (sample_count > 1 时)                      │
│  7. 反归一化: pred = pred * (std + 1e-5) + mean                 │
│  8. 构建结果 DataFrame                                           │
└──────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                          ┌───────────────────┐
                                          │ 原始预测 DataFrame │
                                          │ (raw prediction)  │
                                          └─────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│              apply_price_limits()  [新增]                         │
│                                                                  │
│  A股涨跌停后处理 (±10%):                                        │
│  for each row:                                                   │
│    limit_up   = prev_close × (1 + limit_rate)                   │
│    limit_down = prev_close × (1 - limit_rate)                   │
│    open/high/low/close = clip(value, limit_down, limit_up)      │
│    prev_close = close (更新为当前收盘价)                          │
└──────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                          ┌───────────────────┐
                                          │ 后处理预测结果     │
                                          │ (post-processed)  │
                                          └─────────┬─────────┘
                                                     │
                                          ┌──────────┴──────────┐
                                          │ LLM 已配置?          │
                                          └──────────┬──────────┘
                                              No     │     Yes
                                              │      │      │
                                              │      ▼      │
┌──────────────────────────────────────────────────────────────────┐
│           LLMAnalyzer.analyze_prediction()  [新增]               │
│                                                                  │
│  1. 构建分析 Prompt (包含预测摘要 + 前10期数据)                  │
│  2. 调用 LLM API (通过 litellm)                                  │
│  3. 解析 JSON 响应，提取:                                        │
│     • trend: 趋势判断 (bullish/bearish/range-bound)              │
│     • support_resistance: 支撑/阻力位                            │
│     • advice: 投资建议 (buy/sell/hold)                           │
│     • risk: 风险提示                                             │
└──────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                          ┌───────────────────┐
                                          │  分析报告 (可选)    │
                                          │  {trend, support_  │
                                          │   resistance,      │
                                          │   advice, risk}    │
                                          └─────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                  API 响应 JSON  [新增]                            │
│                                                                  │
│  {                                                               │
│    "success": true,                                              │
│    "prediction": [                                               │
│      {"open": ..., "high": ..., "low": ..., "close": ...,       │
│       "volume": ..., "amount": ...},                             │
│      ...                                                         │
│    ],                                                            │
│    "analysis": {           // 可选，LLM 已配置时才有             │
│      "trend": "...",                                             │
│      "support_resistance": "...",                                │
│      "advice": "...",                                            │
│      "risk": "..."                                               │
│    },                                                            │
│    "params": {                                                   │
│      "lookback": 400,                                            │
│      "pred_len": 120,                                            │
│      "temperature": 1.0,                                         │
│      "top_p": 0.9,                                               │
│      "sample_count": 1,                                          │
│      "symbol": "..."                                             │
│    }                                                             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 配置加载流程

```
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐
│  .env    │────▶│  KronosConfig    │────▶│  校验 + 设备检测   │
│  [新增]  │     │  .load_config()  │     │  .validate()      │
│          │     │  [新增]          │     │  .get_device()    │
└──────────┘     └──────────────────┘     └───────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  注入到各模块        │
               │  • api.py → 启动    │
               │  • KronosPredictor  │
               │  • LLMAnalyzer      │
               └─────────────────────┘
```

---

## 6. 设备选择影响

| 特性 | CUDA (GPU) | MPS (Apple Silicon) | CPU |
|------|------------|---------------------|-----|
| **预测速度** | ⚡ 最快 | 🔵 中等 | 🐢 最慢 |
| **结果准确性** | ✅ 可靠 | ⚠️ `scaled_dot_product_attention` 已知精度问题 | ✅ 可靠 |
| **跨设备可复现** | ❌ 不同 RNG 状态 | ❌ 不同 RNG 状态 | ❌ 不同 RNG 状态 |
| **显存/内存** | 受 GPU 显存限制（H20: ~80GB） | 共享系统统一内存 | 最充裕，仅受系统 RAM 限制 |
| **BF16 支持** | ✅ 将来可启用 | ⚠️ 部分支持 | ✅ 将来可启用 |
| **适用场景** | 生产环境、批量推理 | Mac 开发/调试 | 开发调试、无 GPU 环境 |
| **当前硬件** | 8× NVIDIA H20 (NVSwitch) | — | 通用服务器 CPU |

### 设备自动检测逻辑

系统按以下优先级自动选择设备（`config.py` 中 `get_device()` 方法）：

```
1. CUDA  →  torch.cuda.is_available() == True  →  "cuda:0"
2. MPS   →  torch.backends.mps.is_available() == True  →  "mps"
3. CPU   →  兜底  →  "cpu"
```

也可通过 `.env` 中 `KRONOS_DEVICE` 手动指定设备（如 `cuda:1`、`cpu`），设置为 `auto` 时使用上述自动检测逻辑。

### 注意事项

- **MPS 精度问题**: PyTorch 在 MPS 设备上使用 `scaled_dot_product_attention` 时存在已知精度问题，可能导致预测结果与 CUDA/CPU 不一致。如需精确结果，建议在 MPS 环境下强制使用 CPU（设置 `KRONOS_DEVICE=cpu`）。
- **跨设备不可复现**: 即使使用相同种子，不同设备上的随机数生成器（RNG）实现不同，因此预测结果不可跨设备复现。同一设备上设置相同种子可保证可复现性。
- **GPU 显存管理**: Kronos-small (~4GB) 和 Kronos-base (~8GB) 在 H20 上可轻松运行；Kronos-large (~499M 参数) 需更大显存，暂未开放。

---

## 7. 模块 API 参考

### 7.1 config.py — 配置加载

```python
from config import KronosConfig

cfg = KronosConfig.load_config()                       # 从 .env 加载
cfg = KronosConfig.load_config(env_path="/path/.env")   # 指定路径
cfg = KronosConfig(kronos_model="NeoQuasar/Kronos-base") # 直接构造

cfg.get_device()         # → "cuda:0" / "mps" / "cpu"
cfg.is_llm_configured()  # → True / False
cfg.validate()           # → [] (空=无错误)
```

| 字段 | 类型 | 默认值 | 环境变量 |
|------|------|--------|---------|
| `kronos_model` | str | `NeoQuasar/Kronos-small` | `KRONOS_MODEL` |
| `kronos_tokenizer` | str | `NeoQuasar/Kronos-Tokenizer-base` | `KRONOS_TOKENIZER` |
| `kronos_device` | str | `auto` | `KRONOS_DEVICE` |
| `kronos_max_context` | int | 512 | `KRONOS_MAX_CONTEXT` |
| `kronos_lookback` | int | 400 | `KRONOS_LOOKBACK` |
| `kronos_pred_len` | int | 120 | `KRONOS_PRED_LEN` |
| `kronos_temperature` | float | 1.0 | `KRONOS_TEMPERATURE` |
| `kronos_top_p` | float | 0.9 | `KRONOS_TOP_P` |
| `kronos_sample_count` | int | 1 | `KRONOS_SAMPLE_COUNT` |
| `llm_api_base` | str | `https://open.bigmodel.cn/api/paas/v4` | `LLM_API_BASE` |
| `llm_model_id` | str | `glm-4-flash` | `LLM_MODEL_ID` |
| `llm_api_key` | str | `""` | `LLM_API_KEY` |
| `api_host` | str | `0.0.0.0` | `API_HOST` |
| `api_port` | int | 8000 | `API_PORT` |
| `qlib_provider_uri` | str | `/data02/home/zxh/qlib_local_data/cn_data` | `QLIB_PROVIDER_URI` |
| `max_batch_size` | int | 50 | `MAX_BATCH_SIZE` |

### 7.2 data_loader.py — 数据加载

```python
from data_loader import load_csv, load_qlib, apply_price_limits

df = load_csv("data.csv")
# → DataFrame [timestamps, open, high, low, close, volume, amount]
# 必须列: open, high, low, close, volume
# 可选列: amount (缺失填充 0.0)
# 异常: ValueError (缺少必须列)

df = load_qlib(symbol="SH600977", start_time="2024-01-01", end_time="2025-05-16")
# → DataFrame [timestamps, open, high, low, close, volume, amount]
# 异常: ValueError (qlib 空数据), ImportError (pyqlib 未安装)

pred_df = apply_price_limits(pred_df, last_close=10.0, limit_rate=0.1)
# → 处理后的 DataFrame，open/high/low/close 裁剪到 ±limit_rate
```

### 7.3 llm_analyzer.py — LLM 分析

```python
from llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(
    api_base="https://open.bigmodel.cn/api/paas/v4",
    model_id="glm-4-flash",
    api_key="your-api-key",
)

analyzer.is_available()  # → True / False

result = analyzer.analyze_prediction(pred_df, last_close=10.0, symbol="SH600977")
# → {"trend": "...", "support_resistance": "...", "advice": "...", "risk": "..."}
# → None (LLM 不可用 / 调用失败)
```

| 提供商 | api_base | model_id 示例 |
|--------|----------|--------------|
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 火山引擎 | `https://ark.cn-beijing.volces.com/api/v3` | CodingPlan 模型 ID |

### 7.4 api.py — REST API

启动：`uvicorn api:app --host 0.0.0.0 --port 8000`

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /api/health` | GET | 健康检查 |
| `GET /api/model-status` | GET | 模型状态 |
| `POST /api/predict` | POST | 上传 CSV 预测 |
| `POST /api/predict-qlib` | POST | 从 qlib 读取数据预测 |

`POST /api/predict` 参数：`file`(File,必填), `symbol`, `lookback`, `pred_len`, `temperature`, `top_p`, `sample_count`

`POST /api/predict-qlib` 参数：`symbol`(str,必填), `start_time`, `end_time`, `lookback`, `pred_len`, `temperature`, `top_p`, `sample_count`

响应格式：
```json
{
  "success": true,
  "prediction": [{"open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "amount": ...}],
  "analysis": {"trend": "...", "support_resistance": "...", "advice": "...", "risk": "..."},
  "params": {"lookback": 400, "pred_len": 120, "temperature": 1.0, "top_p": 0.9, "sample_count": 1, "symbol": "..."}
}
```

错误码：200(成功), 400(数据问题), 503(模型/pyqlib 未就绪)

跨环境调用（推荐，避免依赖冲突）：
```python
import requests
resp = requests.post("http://localhost:8000/api/predict-qlib", data={
    "symbol": "SH600977", "start_time": "2024-01-01", "end_time": "2025-05-16",
})
result = resp.json()
```

---

## 附录：项目目录结构

```
kronos/
├── .trae/
│   └── rules/
│       └── kronos_rules.md          # 项目开发规则
├── model/                           # [自带] Kronos 核心模型代码（勿修改）
│   ├── __init__.py                  # 导出 KronosTokenizer, Kronos, KronosPredictor
│   ├── kronos.py                    # 模型定义 + 分词器 + 预测器 + 推理引擎
│   └── module.py                    # Transformer 组件（注意力、前馈、归一化等）
├── finetune/                        # [自带] Qlib 数据微调
│   ├── train_predictor.py
│   ├── train_tokenizer.py
│   ├── dataset.py
│   ├── config.py
│   ├── qlib_data_preprocess.py
│   ├── qlib_test.py
│   └── utils/
├── finetune_csv/                    # [自带] CSV 数据微调
│   ├── finetune_tokenizer.py
│   ├── finetune_base_model.py
│   ├── train_sequential.py
│   ├── config_loader.py
│   ├── configs/
│   ├── data/
│   └── examples/
├── examples/                        # [自带] 预测示例
│   ├── prediction_example.py
│   ├── prediction_wo_vol_example.py
│   ├── prediction_batch_example.py
│   ├── prediction_cn_markets_day.py
│   ├── prediction_akshare_2024-2025.py
│   ├── prediction_new.py
│   ├── prediction_new_GUI.py
│   ├── get_akshare_date_2024-2025_x.py
│   ├── get_date_new.py
│   ├── run_backtest_kronos.py
│   └── yuce/
├── webui/                           # [自带] Web UI（Flask，端口 7070）
│   ├── app.py
│   ├── run.py
│   ├── start.sh
│   ├── requirements.txt
│   ├── templates/
│   └── prediction_results/
├── config.py                        # [新增] 统一配置管理
├── data_loader.py                   # [新增] 数据加载器
├── llm_analyzer.py                  # [新增] LLM 分析器
├── api.py                           # [新增] FastAPI 推理服务
├── .env                             # [新增] 配置文件（勿提交）
├── .env.example                     # [新增] 配置示例
├── requirements.txt
└── figures/
```
