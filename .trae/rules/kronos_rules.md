# Kronos 项目开发规则

## 环境配置

### 1. 虚拟环境管理
- **环境管理工具**: conda
- **虚拟环境名称**: kronos
- **Python 版本**: 3.10

### 2. 创建 conda 环境
```bash
conda create -n kronos python=3.10 -y
conda activate kronos
```

### 3. 安装依赖
```bash
cd /home/zxh/quant_projects/kronos
pip install -r requirements.txt
```

### 4. 当前环境依赖版本

| 依赖 | 已安装版本 | 说明 |
|------|-----------|------|
| Python | 3.10.20 | conda 管理 |
| numpy | 2.2.6 | |
| pandas | 2.2.2 | |
| torch | 2.5.1+cu121 | CUDA 12.1，兼容驱动 535.161.08 |
| einops | 0.8.1 | |
| huggingface_hub | 0.33.1 | |
| matplotlib | 3.9.3 | |
| tqdm | 4.67.1 | |
| safetensors | 0.6.2 | |

### 5. 导出环境配置
```bash
conda env export > environment.yml
conda env create -f environment.yml
```

## GPU 推理环境

### 硬件信息
- **GPU**: 8 块 NVIDIA H20（NVSwitch 互联）
- **NVIDIA 驱动**: 535.161.08（支持最高 CUDA 12.2）
- **PyTorch**: 2.5.1+cu121（已降级，兼容驱动 535）

### 设备自动检测逻辑
```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### 注意事项
- Trae IDE 沙箱无 GPU 直通，GPU 验证需在 SSH 终端执行
- 如遇 PyTorch CUDA 版本与驱动不兼容，需降级 PyTorch（参考当前已安装版本）

## Kronos 使用方法

### 1. 环境准备

```bash
conda activate kronos
cd /home/zxh/quant_projects/kronos
```

Kronos 尚未支持 `pip install`，需通过 `sys.path` 导入：
```python
import sys
sys.path.insert(0, "/home/zxh/quant_projects/kronos")
from model import Kronos, KronosTokenizer, KronosPredictor
```

### 2. 可用模型

| 模型 | 分词器 | 上下文长度 | 参数量 | 显存需求 | Hugging Face 地址 | 适用场景 |
|------|--------|-----------|--------|---------|-------------------|---------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | ~2GB | NeoQuasar/Kronos-mini | 实时监控、低配设备 |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | ~4GB | NeoQuasar/Kronos-small | 个人投资、日常分析（推荐入门） |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | ~8GB | NeoQuasar/Kronos-base | 专业交易、机构使用 |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | 暂未开放 | 暂未开放 | - |

### 2.1 模型主要作用

Kronos 是一个**基于 Token 化的自回归时间序列预测模型**，核心作用是将金融 K 线数据转化为离散 token 序列，然后用 Transformer 进行自回归预测。与传统时间序列模型不同，Kronos 借鉴了大语言模型（LLM）的架构思想：

1. **Tokenizer（分词器）**：将连续的 OHLCV+A 数值序列编码为离散 token ID
2. **Kronos Model（预测模型）**：基于 token 序列进行自回归生成，预测未来 token
3. **解码**：将预测的 token ID 解码回 OHLCV+A 数值

### 2.2 模型架构分类

Kronos 系统由三个核心组件构成：

| 组件 | 类名 | 作用 | 源码位置 |
|------|------|------|---------|
| **分词器** | `KronosTokenizer` | 将连续数值编码为离散 token，或将 token 解码回数值 | [kronos.py:L13-177](file:///home/zxh/quant_projects/kronos/model/kronos.py#L13-177) |
| **预测模型** | `Kronos` | 基于历史 token 自回归预测未来 token | [kronos.py:L180-296](file:///home/zxh/quant_projects/kronos/model/kronos.py#L180-296) |
| **预测器** | `KronosPredictor` | 封装完整的预测流程（归一化→编码→推理→解码→反归一化） | [kronos.py:L482-559](file:///home/zxh/quant_projects/kronos/model/kronos.py#L482-559) |

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

### 3. 加载模型

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

`KronosPredictor.__init__` 参数：
- `model`: Kronos 模型实例
- `tokenizer`: KronosTokenizer 实例
- `device`: 设备，默认自动检测（CUDA → MPS → CPU）
- `max_context`: 上下文窗口长度，默认 512（mini 模型用 2048）
- `clip`: 归一化裁剪范围，默认 5

### 4. 数据格式要求

- **必须列**: `open`, `high`, `low`, `close`, `volume`（缺失会抛出 ValueError）
- **可选列**: `amount`（缺失时自动估算或填充 0.0）
- **时间列**: `timestamps` 或 `date`（需为 datetime 类型）
- **推荐 lookback**: 400（small/base 模型 max_context=512）
- **推荐 pred_len**: 20-120
- **模型内部维度**: 始终为 6 维 `[open, high, low, close, volume, amount]`，`volume` 为必填，`amount` 缺失时自动补全

### 5. 单序列预测 — `predict()`

```python
import pandas as pd

df = pd.read_csv("data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[:lookback]['timestamps']
y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

`predict()` 参数：
- `df`: DataFrame，包含 OHLCV 列
- `x_timestamp`: 历史时间戳（pd.DatetimeIndex 或 Series）
- `y_timestamp`: 未来时间戳（长度需等于 pred_len）
- `pred_len`: 预测步数
- `T`: 采样温度，默认 1.0；降低（如 0.6）得到更保守的预测
- `top_k`: Top-k 过滤，默认 0（不启用）
- `top_p`: 核采样阈值，默认 0.9
- `sample_count`: 采样次数，增大可得到概率预测（多条路径取平均）
- `verbose`: 是否显示自回归进度条

返回：DataFrame，列为 `[open, high, low, close, volume, amount]`，索引为 `y_timestamp`

#### 仅提供 OHLC 的最简用法

```python
x_df = df.iloc[:lookback][['open', 'high', 'low', 'close']]

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
)
```

`volume` 和 `amount` 缺失时自动填充 0.0。

### 6. 批量预测 — `predict_batch()`

对多条时间序列并行预测，所有序列需具有相同的历史长度和预测长度：

```python
df_list = []
x_timestamp_list = []
y_timestamp_list = []

for i in range(batch_size):
    idf = df.iloc[i*400:(i+1)*400+pred_len]
    x_df = idf.iloc[:400][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts = idf.iloc[:400]['timestamps']
    y_ts = idf.iloc[400:400+pred_len]['timestamps']
    df_list.append(x_df)
    x_timestamp_list.append(x_ts)
    y_timestamp_list.append(y_ts)

pred_dfs = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

返回：`List[DataFrame]`，每个 DataFrame 格式同 `predict()`

### 7. 生成未来时间戳

预测未来数据时，`y_timestamp` 需要自行生成：

```python
last_date = df['timestamps'].iloc[-1]
y_timestamp = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1),
    periods=pred_len
)
```

### 8. A 股涨跌停后处理

A 股有 ±10% 涨跌停限制，预测结果需后处理：

```python
def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    pred_df = pred_df.reset_index(drop=True)
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")
    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)
        for col in cols:
            pred_df.at[i, col] = max(min(pred_df.at[i, col], limit_up), limit_down)
        last_close = float(pred_df.at[i, "close"])
    return pred_df

last_close = df['close'].iloc[-1]
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)
```

### 9. 从 qlib 本地目录读取数据并预测

```python
import sys
import pandas as pd
sys.path.insert(0, "/home/zxh/quant_projects/kronos")
from model import Kronos, KronosTokenizer, KronosPredictor
from data_loader import load_qlib, apply_price_limits

df = load_qlib(
    symbol="SH600977",
    start_time="2024-01-01",
    end_time="2025-05-16",
    provider_uri="/data02/home/zxh/qlib_local_data/cn_data",
)

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

lookback = 400
pred_len = 120

x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.iloc[-lookback:]["timestamps"]
y_timestamp = pd.bdate_range(
    start=df["timestamps"].iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len
)

pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=True
)

last_close = df["close"].iloc[-1]
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)
```

qlib 字段自动映射：

| qlib 字段 | Kronos 字段 | 说明 |
|-----------|------------|------|
| `$open` | `open` | 开盘价 |
| `$high` | `high` | 最高价 |
| `$low` | `low` | 最低价 |
| `$close` | `close` | 收盘价 |
| `$volume` | `volume` | 成交量 |
| `$amount` | `amount` | 成交额 |
| DataFrame 索引 | `timestamps` | 日期时间 |

依赖：`pip install pyqlib`

### 10. 完整预测流程示例

```python
import sys
import pandas as pd
sys.path.insert(0, "/home/zxh/quant_projects/kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv("data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[-lookback:]['timestamps']
y_timestamp = pd.bdate_range(
    start=df['timestamps'].iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len
)

pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=True
)

last_close = df['close'].iloc[-1]
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

print(pred_df.head())
```

### 11. 模块 API 参考

#### config.py — 配置加载

```python
from config import KronosConfig

cfg = KronosConfig.load_config()                    # 从 .env 加载
cfg = KronosConfig.load_config(env_path="/path/.env")  # 指定路径
cfg = KronosConfig(kronos_model="NeoQuasar/Kronos-base", llm_api_key="key")  # 直接构造

cfg.get_device()         # → "cuda:0" / "mps" / "cpu"
cfg.is_llm_configured()  # → True / False
cfg.validate()           # → [] (空=无错误)
```

配置字段：

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

#### data_loader.py — 数据加载

```python
from data_loader import load_csv, load_qlib, apply_price_limits

# CSV 加载（必须列: open, high, low, close, volume）
df = load_csv("data.csv")
# → DataFrame [timestamps, open, high, low, close, volume, amount]

# qlib 加载
df = load_qlib(symbol="SH600977", start_time="2024-01-01", end_time="2025-05-16")
# → DataFrame [timestamps, open, high, low, close, volume, amount]

# A 股涨跌停后处理
pred_df = apply_price_limits(pred_df, last_close=10.0, limit_rate=0.1)
```

异常：`ValueError`（缺少必须列/qlib 空数据）、`ImportError`（pyqlib 未安装）

#### llm_analyzer.py — LLM 分析

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
# → None（LLM 不可用 / 调用失败）
```

支持的 LLM 提供商：

| 提供商 | api_base | model_id 示例 |
|--------|----------|--------------|
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 火山引擎 | `https://ark.cn-beijing.volces.com/api/v3` | CodingPlan 模型 ID |

### 12. REST API 参考

启动服务：`uvicorn api:app --host 0.0.0.0 --port 8000`

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /api/health` | GET | 健康检查 |
| `GET /api/model-status` | GET | 模型状态 |
| `POST /api/predict` | POST | 上传 CSV 预测 |
| `POST /api/predict-qlib` | POST | 从 qlib 读取数据预测 |

#### 请求参数

`POST /api/predict`：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | CSV 文件 |
| `symbol` | str | ❌ | None | 股票代码 |
| `lookback` | int | ❌ | 400 | 历史长度 |
| `pred_len` | int | ❌ | 120 | 预测步数 |
| `temperature` | float | ❌ | 1.0 | 采样温度 |
| `top_p` | float | ❌ | 0.9 | 核采样阈值 |
| `sample_count` | int | ❌ | 1 | 采样次数 |

`POST /api/predict-qlib`：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `symbol` | str | ✅ | - | 股票代码 |
| `start_time` | str | ❌ | 2024-01-01 | 开始日期 |
| `end_time` | str | ❌ | 2025-05-16 | 结束日期 |
| `lookback` | int | ❌ | 400 | 历史长度 |
| `pred_len` | int | ❌ | 120 | 预测步数 |
| `temperature` | float | ❌ | 1.0 | 采样温度 |
| `top_p` | float | ❌ | 0.9 | 核采样阈值 |
| `sample_count` | int | ❌ | 1 | 采样次数 |

#### 响应格式

```json
{
  "success": true,
  "prediction": [
    {"open": 10.23, "high": 10.32, "low": 10.19, "close": 10.30, "volume": 8797436.0, "amount": 90268096.0}
  ],
  "analysis": {
    "trend": "趋势解读",
    "support_resistance": "支撑阻力位",
    "advice": "投资建议",
    "risk": "风险提示"
  },
  "params": {"lookback": 400, "pred_len": 120, "temperature": 1.0, "top_p": 0.9, "sample_count": 1, "symbol": "SH600977"}
}
```

#### 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 数据不足 / 缺少必须列 / qlib 返回空 |
| 503 | 模型未加载 / pyqlib 未安装 |

#### 调用示例

```bash
# CSV 预测
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data.csv" -F "symbol=SH600977" -F "pred_len=60"

# qlib 预测
curl -X POST http://localhost:8000/api/predict-qlib \
  -F "symbol=SH600977" -F "start_time=2024-01-01" -F "end_time=2025-05-16"
```

#### 跨环境调用

其他 conda 环境（如 qlib）通过 REST API 调用（推荐，避免依赖冲突）：

```python
import requests
resp = requests.post("http://localhost:8000/api/predict-qlib", data={
    "symbol": "SH600977", "start_time": "2024-01-01", "end_time": "2025-05-16",
})
result = resp.json()
```

### 核心原则
- **不修改** `model/` 目录下的核心模型代码
- **不影响** 其他 conda 环境（rdagent/qlib）
- 新增功能模块放在项目根目录

### 项目结构约定
```
kronos/
├── .trae/
│   └── rules/
│       └── kronos_rules.md
├── model/                        # Kronos 核心模型代码（勿修改）
│   ├── __init__.py
│   ├── kronos.py
│   └── module.py
├── finetune/                     # Qlib 数据微调
├── finetune_csv/                 # CSV 数据微调
├── examples/                     # 预测示例
├── webui/                        # Web UI（Flask，端口 7070）
├── tests/
├── requirements.txt
├── .env                          # 统一配置（勿提交）
├── .env.example                  # 配置示例
├── config.py                     # 配置加载模块
├── data_loader.py                # 数据源模块（CSV）
├── llm_analyzer.py               # LLM 分析器
└── api.py                        # FastAPI 推理服务
```

## 磁盘空间

### 当前磁盘状态
- `/dev/vda2` (根目录): 约 6.6G 可用
- `/data01` (大容量盘): 1.3T 可用

### 空间优化
- 使用 `--no-cache-dir` 参数安装 pip 包
- conda 环境位于 `/home/zxh/miniconda3/envs/kronos`
- 定期清理 conda 缓存：`conda clean --all`

## 注意事项

### 兼容性说明
- Kronos 官方推荐 Python 3.10+，使用 Python 3.10 兼容性最佳
- PyTorch 需与 NVIDIA 驱动匹配：驱动 535 → CUDA ≤12.2 → PyTorch cu121

### 依赖管理
- 使用 conda 管理 Python 版本和环境
- 使用 pip 安装 Python 包（在 conda 环境内）
- requirements.txt 中有重复条目（numpy/pandas 各出现两次），已知问题

### 用户权限约束
- **重要**: 本项目使用非 root 账户
- 禁止使用 `sudo` 或其他需要 root 权限的操作
- 如需执行必须的系统级操作，请先明确说明原因并等待用户确认

### 代码规范
- 遵循 PEP 8 代码风格
- 使用类型提示
- 编写文档字符串

### Git 提交规范
- 提交信息应清晰描述完成的工作
- 保持提交历史的整洁和可读性
- 提交前检查是否有敏感信息（如 .env 文件已在 .gitignore 中）

### Hugging Face 下载
- 模型权重从 Hugging Face Hub 下载，需确保网络可访问
- 如不可访问需配置镜像：`HF_ENDPOINT=https://hf-mirror.com`
