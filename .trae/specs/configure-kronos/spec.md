# Kronos 配置 Spec

## Why
Kronos 已安装完成，GPU 推理环境已修复（PyTorch 2.5.1+cu121）。需要完成 Kronos 的应用层封装：统一配置管理、LLM 增强分析、FastAPI 推理服务。

## What Changes
- 创建统一配置文件（`.env`），管理模型、设备、LLM 等参数
- 新增 LLM 分析器，用 LLM 对预测结果进行解读和生成投资建议
- 创建数据源模块，支持 CSV 文件加载
- 创建 FastAPI 推理服务，提供 REST API 接口

## Impact
- Affected code: `kronos/` 项目根目录新增配置文件和模块
- Affected deps: 安装 litellm/python-dotenv/fastapi/uvicorn
- 不修改 Kronos 核心模型代码（`model/` 目录）
- 不影响其他 conda 环境（rdagent/qlib）

## ADDED Requirements

### Requirement: 统一配置管理
系统 SHALL 提供统一的配置文件（`.env`）管理所有参数。

#### Scenario: 配置加载
- **WHEN** 用户启动预测脚本或 API 服务
- **THEN** 系统从 `.env` 文件加载配置

#### Scenario: 配置项
配置文件 SHALL 包含以下配置项：
```
# Kronos 模型配置
KRONOS_MODEL=NeoQuasar/Kronos-small
KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
KRONOS_DEVICE=auto
KRONOS_MAX_CONTEXT=512

# 预测参数
# LOOKBACK: 输入历史数据长度（K 线根数），模型基于此窗口进行预测
#   - small/base 模型 max_context=512，推荐 400（留余量给 pred_len）
#   - mini 模型 max_context=2048，可用更大值
#   - 低于 400 预测质量下降，高于 max_context-pred_len 会被截断
KRONOS_LOOKBACK=400
# PRED_LEN: 预测未来 K 线根数，即自回归解码步数
#   - 推荐 20-120，步数越多累积误差越大
#   - 需配合 y_timestamp 长度一致
KRONOS_PRED_LEN=120
# TEMPERATURE: 采样温度，控制预测随机性
#   - T=1.0 标准采样，反映模型学到的分布
#   - T<1.0（如 0.6）更保守，倾向于高概率 token，预测更稳定
#   - T>1.0（如 1.5）更随机，探索更多可能性，预测波动更大
#   - T→0 退化为贪心解码（取概率最高的 token）
KRONOS_TEMPERATURE=1.0
# TOP_P: 核采样阈值（Nucleus Sampling），从累积概率 ≥ top_p 的最小 token 集合中采样
#   - top_p=0.9 保留累积概率前 90% 的 token，过滤掉尾部低概率项
#   - top_p=1.0 不做过滤，从全部 token 中采样
#   - 越小越保守，越大越多样
KRONOS_TOP_P=0.9
# SAMPLE_COUNT: 并行采样次数，生成多条预测路径
#   - sample_count=1 单次采样，输出一条预测
#   - sample_count>1 多次采样，输出取均值（概率预测）
#   - 增大可得到置信区间，但显存和耗时线性增长
KRONOS_SAMPLE_COUNT=1

# LLM 配置（支持火山引擎 CodingPlan / 智谱 CodingPlan 等）
# 火山引擎: LLM_API_BASE=https://ark.cn-beijing.volces.com/api/v3
# 智谱:     LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL_ID=glm-4-flash
LLM_API_KEY=your_api_key_here

# API 服务配置
API_HOST=0.0.0.0
API_PORT=8000
```

#### Scenario: 设备自动检测
- **WHEN** `KRONOS_DEVICE=auto`
- **THEN** 系统自动检测：CUDA 可用 → cuda:0；MPS 可用 → mps；否则 → cpu

### Requirement: LLM 增强分析
系统 SHALL 支持使用 custom LLM 对 Kronos 预测结果进行智能分析，通过 litellm 统一调用不同 LLM 提供商。

#### Scenario: LLM 分析预测结果
- **WHEN** Kronos 完成价格预测后且 LLM 已配置
- **THEN** 系统将预测结果构建为结构化 prompt，发送给 LLM，生成分析报告（趋势解读、支撑阻力、投资建议、风险提示）

#### Scenario: LLM 不可用时回退
- **WHEN** LLM API 不可用或未配置（API key 为空）
- **THEN** 系统仍能完成 Kronos 预测，跳过 LLM 分析，输出原始预测结果

#### Scenario: LLM 提供商兼容性
- **WHEN** 用户配置不同的 LLM 提供商
- **THEN** 系统通过 litellm 统一接口调用，只需修改 base URL、model ID 和 API key

#### Scenario: 火山引擎（Volcengine）CodingPlan
- **WHEN** 用户配置 LLM_API_BASE 为火山引擎端点（如 `https://ark.cn-beijing.volces.com/api/v3`）且 LLM_MODEL_ID 为 CodingPlan 模型
- **THEN** 系统通过 litellm 以 OpenAI 兼容模式调用火山引擎 CodingPlan

#### Scenario: 智谱（BigModel）CodingPlan
- **WHEN** 用户配置 LLM_API_BASE 为智谱端点（如 `https://open.bigmodel.cn/api/paas/v4`）且 LLM_MODEL_ID 为 CodingPlan 模型
- **THEN** 系统通过 litellm 以 OpenAI 兼容模式调用智谱 CodingPlan

### Requirement: FastAPI 推理服务
系统 SHALL 提供 FastAPI 推理服务，支持通过 REST API 进行预测分析。

#### Scenario: 启动服务
- **WHEN** 用户运行 `python api.py` 或 `uvicorn api:app`
- **THEN** 系统启动 FastAPI 服务，加载 Kronos 模型，监听配置的端口

#### Scenario: API 预测接口
- **WHEN** 客户端发送 `POST /api/predict` 请求（包含股票代码或数据）
- **THEN** 系统执行预测并返回 JSON 结果（包含预测数据和可选的 LLM 分析）

#### Scenario: API 端点
服务 SHALL 提供以下端点：
- `POST /api/predict` — 执行预测（支持 csv 数据上传）
- `GET /api/model-status` — 查看模型加载状态
- `GET /api/health` — 健康检查

### Requirement: 数据源支持
系统 SHALL 支持 CSV 文件数据源，支持自动字段映射和 A 股涨跌停后处理。
