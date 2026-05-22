# Volume 必填项 + Qlib 本地数据源 Spec

## Why
当前 `volume` 为可选项，缺失时填充 0.0，导致模型在无成交量数据时预测质量下降。成交量是重要的市场信号，应在预测过程中必须考虑 volume 因素。同时，服务器上已有 qlib 本地数据目录（`/data02/home/zxh/qlib_local_data/cn_data`），包含 OHLCV+A 数据，应支持直接从 qlib 目录读取。

## What Changes
- 将 `volume` 从可选列提升为必填列，缺失时抛出 ValueError
- 在 `data_loader.py` 中新增 `load_qlib()` 函数，支持从 qlib 本地目录读取数据
- 更新 `api.py` 新增 `POST /api/predict-qlib` 端点
- 更新 `.env` 新增 qlib 数据路径配置
- 更新 `kronos_rules.md` 和 `design.md` 中的数据格式说明
- `amount` 仍为可选（缺失时自动估算）

## Impact
- Affected code: `data_loader.py`, `api.py`, `config.py`, `.env`, `.env.example`
- Affected docs: `kronos_rules.md`, `design.md`
- **BREAKING**: 之前不含 `volume` 列的 CSV 文件将无法加载，需补充成交量数据
- qlib 依赖仅在 `qlib` conda 环境中可用，`kronos` 环境需安装 pyqlib

## ADDED Requirements

### Requirement: Qlib 本地数据源
系统 SHALL 支持从 qlib 本地数据目录直接读取 K 线数据。

#### Scenario: 从 qlib 读取单只股票
- **WHEN** 用户指定股票代码（如 `sh600977`）和时间范围
- **THEN** 系统从 qlib 本地目录读取 OHLCV+A 数据，自动转换为 Kronos 所需格式

#### Scenario: qlib 数据字段映射
- **WHEN** 从 qlib 读取数据
- **THEN** 系统自动完成字段映射：`$open→open, $high→high, $low→low, $close→close, $volume→volume, $amount→amount`，索引 `datetime→timestamps`

#### Scenario: qlib 不可用时回退
- **WHEN** qlib 未安装或数据目录不存在
- **THEN** 系统给出明确提示，不影响 CSV 加载功能

#### Scenario: qlib 环境依赖
- **WHEN** 使用 qlib 数据源
- **THEN** 需要安装 pyqlib 包（`pip install pyqlib`）

### Requirement: API 支持 qlib 数据源
系统 SHALL 提供 API 端点支持从 qlib 读取数据并预测。

#### Scenario: qlib 预测端点
- **WHEN** 客户端发送 `POST /api/predict-qlib` 请求（包含 symbol, start_time, end_time）
- **THEN** 系统从 qlib 读取数据，执行预测，返回 JSON 结果

### Requirement: 批量预测股票数量限制
系统 SHALL 明确批量预测的股票数量限制，并在超出限制时给出提示。

#### Scenario: 批量预测数量限制
- **WHEN** 用户通过 `predict_batch()` 或 API 批量预测多只股票
- **THEN** 系统限制单次批量预测的股票数量，防止 GPU 显存溢出

#### Scenario: 批量预测数量约束说明
Kronos 代码中没有硬编码的批量大小限制，实际限制由 GPU 显存决定：
- 批量张量形状：`(B × sample_count, seq_len, 6)`，其中 B = 股票数量
- 每步自回归推理需对整个 batch 做一次前向传播
- 估算（H20 80GB VRAM，Kronos-small ~4GB 模型权重）：
  - `sample_count=1, lookback=400, pred_len=120`: 约 50-100 只股票
  - `sample_count=5`: 约 10-20 只股票
  - 超出显存时 PyTorch 抛出 CUDA OOM 错误
- 建议 API 层设置默认 `max_batch_size=50`，可通过 `.env` 配置

## MODIFIED Requirements

### Requirement: 数据源支持
系统 SHALL 支持 CSV 文件和 qlib 本地目录两种数据源，`volume` 为必填列，`amount` 仍为可选列。

#### Scenario: volume 缺失时拒绝加载
- **WHEN** 用户上传的 CSV 文件缺少 `volume` 列
- **THEN** 系统抛出 ValueError，提示用户必须提供成交量数据

#### Scenario: volume 存在但 amount 缺失
- **WHEN** CSV 文件包含 `volume` 但缺少 `amount`
- **THEN** 系统自动估算 `amount = volume × 均价`（KronosPredictor 内置逻辑）

## REMOVED Requirements

### Requirement: volume 可选填充 0.0
**Reason**: 成交量是重要的市场信号，填充 0.0 会导致模型在无成交量数据时预测质量下降
**Migration**: 用户需在 CSV 中补充 `volume` 列数据，或使用 qlib 数据源（自动包含 volume）
