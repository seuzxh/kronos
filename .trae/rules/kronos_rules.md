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
