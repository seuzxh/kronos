<div align="center">
  <h2><b>Kronos: A Foundation Model for the Language of Financial Markets </b></h2>
</div>


<div align="center">

</a> 
<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://shiyu-coder.github.io/Kronos-demo/"> <img src="https://img.shields.io/badge/🚀-Live_Demo-brightgreen" alt="Live Demo"> </a>
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<div align="center">
  <a href="https://zdoc.app/de/shiyu-coder/Kronos">Deutsch</a> | 
  <a href="https://zdoc.app/es/shiyu-coder/Kronos">Español</a> | 
  <a href="https://zdoc.app/fr/shiyu-coder/Kronos">Français</a> | 
  <a href="https://zdoc.app/ja/shiyu-coder/Kronos">日本語</a> | 
  <a href="https://zdoc.app/ko/shiyu-coder/Kronos">한국어</a> | 
  <a href="https://zdoc.app/pt/shiyu-coder/Kronos">Português</a> | 
  <a href="https://zdoc.app/ru/shiyu-coder/Kronos">Русский</a> | 
  <a href="https://zdoc.app/zh/shiyu-coder/Kronos">中文</a>
</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos 是首个面向金融 K 线（蜡烛图）的**开源基础模型**，在超过 **45 个全球交易所**的数据上完成预训练。

</div>

## 📰 动态
*   🚩 **[2025.11.10]** Kronos 被 AAAI 2026 接收。
*   🚩 **[2025.08.17]** 微调脚本已发布！可使用自有数据对 Kronos 进行适配。
*   🚩 **[2025.08.02]** 论文已在 [arXiv](https://arxiv.org/abs/2508.02739) 上发布！

<p align="center">

## 📜 模型介绍

**Kronos** 是一系列专门为金融市场"语言"——K 线序列——预训练的 decoder-only 基础模型。与通用时序基础模型不同，Kronos 专为处理金融数据独有的高噪声特性而设计。它采用了一种新颖的两阶段框架：
1. 专用分词器首先将连续的多维 K 线数据（OHLCV）量化为**层次化离散 token**。
2. 大型自回归 Transformer 在这些 token 上进行预训练，使其能够作为统一模型服务于多种量化任务。

<p align="center">
    <img src="figures/overview.png" alt="" align="center" width="700px" />
</p>

## ✨ 在线演示
我们搭建了在线演示页面，可视化展示 Kronos 的预测结果。页面展示了 **BTC/USDT** 交易对未来 24 小时的预测。

**👉 [点击访问在线演示](https://shiyu-coder.github.io/Kronos-demo/)**

## 📦 模型 Zoo
我们发布了不同规模的预训练模型，以适应不同的计算资源和应用需求。所有模型均可从 Hugging Face Hub 获取。

| 模型         | 分词器                                                                          | 上下文长度 | 参数量   | 开源                                                                     |
|-------------|---------------------------------------------------------------------------------|-----------|---------|--------------------------------------------------------------------------|
| Kronos-mini | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)     | 2048      | 4.1M    | ✅ [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)  |
| Kronos-small| [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512       | 24.7M   | ✅ [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) |
| Kronos-base | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512       | 102.3M  | ✅ [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)   |
| Kronos-large| [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512       | 499.2M  | ❌                                                                        |

## 🛠️ 工程配置

| 配置项 | 值 |
|--------|-----|
| 当前模型 | Kronos-base（102.3M 参数，Kronos-Tokenizer-base 分词器，上下文长度 512） |
| 数据源路径 | `/home/zxh/qlib_data`（A 股日K行情，5522 只股票，2020-2026） |
| 输出目录 | `/home/zxh/quant_projects/kronos/outputs` |
| 环境配置 | conda `kronos` / Python 3.10 / PyTorch 2.5.1+cu121 |
| GPU 硬件 | 8×NVIDIA H20（NVSwitch 互联），驱动 535.161.08 |
| 批量预测 | `max_batch_size=50` |

## 🆕 新增特性

- **数据源切换**：从 `/data02/home/zxh/qlib_local_data/cn_data` 切换至 `/home/zxh/qlib_data`，新数据源覆盖 A 股 5522 只股票的日K行情（2020-2026）。
- **vwap→amount 适配**：新数据源缺少 `amount` 字段，自动使用 `vwap * volume` 计算，确保模型 6 维输入完整。
- **预测结果持久化**：每次预测自动保存到 `{output_dir}/{YYYY-MM-DD}/{symbol}/` 目录，便于回溯和对比。
- **批量预测优化**：`max_batch_size` 从 30 提升至 50，充分利用 H20 显存，提升批量推理吞吐量。

## 📂 输出目录结构

预测结果按日期和股票代码组织存储：

```
outputs/
└── 2026-05-23/
    ├── SH600977/
    │   ├── prediction.csv      # 预测 OHLCV+A 数据
    │   └── meta.json           # 预测元信息
    └── SH000001/
        ├── prediction.csv
        └── meta.json
```

**文件说明**：

| 文件 | 说明 |
|------|------|
| `prediction.csv` | 预测的 OHLCV+A 数据，列为 `[open, high, low, close, volume, amount]` |
| `meta.json` | 预测元信息，包含以下字段 |

`meta.json` 字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 股票代码 |
| `timestamp` | str | 预测执行时间（ISO 8601 格式） |
| `lookback` | int | 历史回看长度 |
| `pred_len` | int | 预测步数 |
| `temperature` | float | 采样温度 |
| `top_p` | float | 核采样阈值 |
| `sample_count` | int | 采样次数 |

## 🧪 测试验证

| 测试文件 | 说明 |
|---------|------|
| `tests/test_batch_predict.py` | 批量预测验证，测试多股票并行预测的正确性和吞吐量 |
| `tests/test_e2e_flow.py` | 全流程端到端测试，覆盖数据加载→预测→持久化→涨跌停后处理的完整链路 |

**运行方式**：

```bash
conda activate kronos
python tests/test_e2e_flow.py
```

## 📚 关键文档

| 文档 | 说明 |
|------|------|
| [doc/sampling_parameters.md](doc/sampling_parameters.md) | 采样参数详解（Temperature、top_p、top_k、sample_count） |
| [doc/price_limit_rules.md](doc/price_limit_rules.md) | A 股涨跌停规则与自动识别（主板10%、创业板/科创板20%、北交所30%、ST 5%） |

## 🚀 快速开始

### 安装

1. 安装 Python 3.10+，然后安装依赖：

```shell
pip install -r requirements.txt
```

### 📈 进行预测

使用 `KronosPredictor` 类进行预测非常简单。它封装了数据预处理、归一化、预测和反归一化的完整流程，只需几行代码即可从原始数据获得预测结果。

**重要提示**：`Kronos-small` 和 `Kronos-base` 的 `max_context` 为 **512**，这是模型能处理的最大序列长度。建议输入数据长度（即 `lookback`）不超过此限制。`KronosPredictor` 会自动截断超长上下文。

以下是逐步指南。

#### 1. 加载分词器和模型

首先，从 Hugging Face Hub 加载预训练的 Kronos 模型及其对应的分词器。

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
```

#### 2. 初始化预测器

创建 `KronosPredictor` 实例，传入模型、分词器和设备。

```python
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

#### 3. 准备输入数据

`predict` 方法需要三个主要输入：
- `df`：包含历史 K 线数据的 pandas DataFrame，必须包含 `['open', 'high', 'low', 'close']` 列，`volume` 和 `amount` 为可选列。
- `x_timestamp`：对应历史数据的时间戳（pandas Series）。
- `y_timestamp`：需要预测的未来时间戳（pandas Series）。

```python
import pandas as pd

df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

#### 4. 生成预测

调用 `predict` 方法生成预测。可以通过 `T`、`top_p` 和 `sample_count` 等参数控制采样过程，实现概率预测。

```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1
)

print("预测数据前几行：")
print(pred_df.head())
```

`predict` 方法返回一个 pandas DataFrame，包含 `open`、`high`、`low`、`close`、`volume` 和 `amount` 的预测值，索引为提供的 `y_timestamp`。

对于多条时间序列的高效处理，Kronos 提供了 `predict_batch` 方法，支持多数据集并行预测，特别适合同时预测多个资产或时间段。

```python
df_list = [df1, df2, df3]
x_timestamp_list = [x_ts1, x_ts2, x_ts3]
y_timestamp_list = [y_ts1, y_ts2, y_ts3]

pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

for i, pred_df in enumerate(pred_df_list):
    print(f"序列 {i} 的预测结果：")
    print(pred_df.head())
```

**批量预测要求**：
- 所有序列必须具有相同的历史长度（lookback 窗口）
- 所有序列必须具有相同的预测长度（`pred_len`）
- 每个 DataFrame 必须包含必需列：`['open', 'high', 'low', 'close']`
- `volume` 和 `amount` 列为可选，缺失时自动填充 0

`predict_batch` 方法利用 GPU 并行处理，并自动为每条序列独立处理归一化和反归一化。

#### 5. 示例与可视化

完整可运行的脚本（包含数据加载、预测和绘图）请参见 [`examples/prediction_example.py`](examples/prediction_example.py)。

运行该脚本将生成一张对比图，展示真实数据与模型预测的对比：

<p align="center">
    <img src="figures/prediction_example.png" alt="预测示例" align="center" width="600px" />
</p>

此外，我们还提供了不含成交量和成交额数据的预测脚本，参见 [`examples/prediction_wo_vol_example.py`](examples/prediction_wo_vol_example.py)。


## 🔧 微调指南（A 股市场示例）

我们提供了完整的微调流程，可使用自有数据对 Kronos 进行微调。以下以使用 [Qlib](https://github.com/microsoft/qlib) 准备中国 A 股市场数据并进行简单回测为例。

> **免责声明**：此流程仅用于演示微调过程，是一个简化示例，并非生产级量化交易系统。稳健的量化策略需要更复杂的技术，如投资组合优化和风险因子中性化，才能实现稳定的 alpha。

微调过程分为四个主要步骤：

1. **配置**：设置路径和超参数。
2. **数据准备**：使用 Qlib 处理和切分数据。
3. **模型微调**：微调分词器和预测模型。
4. **回测**：评估微调后模型的性能。

### 前置条件

1. 确保已安装 `requirements.txt` 中的所有依赖。
2. 本流程依赖 `qlib`，请安装：
    ```shell
    pip install pyqlib
    ```
3. 需要准备 Qlib 数据，请按照 [Qlib 官方指南](https://github.com/microsoft/qlib) 下载并在本地配置数据。示例脚本假定使用日频数据。

### 第 1 步：配置实验

数据、训练和模型路径的所有设置集中在 `finetune/config.py` 中。运行任何脚本前，请**根据你的环境修改以下路径**：

* `qlib_data_path`：本地 Qlib 数据目录路径。
* `dataset_path`：处理后的训练/验证/测试 pickle 文件保存目录。
* `save_path`：模型检查点保存的基础目录。
* `backtest_result_path`：回测结果保存目录。
* `pretrained_tokenizer_path` 和 `pretrained_predictor_path`：预训练模型路径（可以是本地路径或 Hugging Face 模型名称）。

还可以调整 `instrument`、`train_time_range`、`epochs`、`batch_size` 等参数。如不使用 [Comet.ml](https://www.comet.com/)，请设置 `use_comet = False`。

### 第 2 步：准备数据集

运行数据预处理脚本。该脚本将从 Qlib 目录加载原始行情数据，处理后切分为训练集、验证集和测试集，并保存为 pickle 文件。

```shell
python finetune/qlib_data_preprocess.py
```

运行后，将在 `config.py` 中 `dataset_path` 指定的目录下生成 `train_data.pkl`、`val_data.pkl` 和 `test_data.pkl`。

### 第 3 步：运行微调

微调过程包含两个阶段：先微调分词器，再微调预测模型。两个训练脚本均支持使用 `torchrun` 进行多 GPU 训练。

#### 3.1 微调分词器

此步骤将分词器调整到特定领域的数据分布。

```shell
# 将 NUM_GPUS 替换为要使用的 GPU 数量（如 2）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

最佳分词器检查点将保存到 `config.py` 中配置的路径。

#### 3.2 微调预测模型

此步骤微调 Kronos 主模型以完成预测任务。

```shell
# 将 NUM_GPUS 替换为要使用的 GPU 数量（如 2）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

最佳预测模型检查点将保存到 `config.py` 中配置的路径。

### 第 4 步：回测评估

最后，运行回测脚本评估微调后的模型。该脚本加载模型，在测试集上进行推理，生成预测信号（如预测价格变化），并运行简单的 Top-K 策略回测。

```shell
python finetune/qlib_test.py --device cuda:0
```

脚本将在控制台输出详细的性能分析，并生成策略累计收益曲线与基准对比图：

<p align="center">
    <img src="figures/backtest_result_example.png" alt="回测示例" align="center" width="700px" />
</p>

### 💡 从演示到生产：重要考量

* **原始信号 vs 纯 Alpha**：本演示中模型生成的信号是原始预测。在实际量化工作流中，这些信号通常会输入投资组合优化模型，通过约束条件中性化常见风险因子（如市场 Beta、规模和价值等风格因子）的暴露，从而提取**"纯 Alpha"**，提升策略稳健性。
* **数据处理**：提供的 `QlibDataset` 是示例。对于不同的数据源或格式，需要适配数据加载和预处理逻辑。
* **策略与回测复杂度**：此处使用的简单 Top-K 策略仅为起点。生产级策略通常包含更复杂的投资组合构建、动态仓位管理和风险管理（如止损/止盈规则）逻辑。此外，高保真回测应细致建模交易成本、滑点和市场冲击，以更准确地估计实际表现。

> **📝 AI 生成注释**：请注意，`finetune/` 目录中的许多代码注释由 AI 助手（Gemini 2.5 Pro）生成，用于解释说明。虽然力求准确，但可能存在不准确之处。建议以代码本身作为逻辑的权威来源。

## 📖 引用

如果在研究中使用 Kronos，请引用我们的[论文](https://arxiv.org/abs/2508.02739)：

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## 📜 许可证
本项目基于 [MIT License](./LICENSE) 许可。
