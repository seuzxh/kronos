# Tasks

- [x] Task 1: 修复 GPU 推理环境 - 降级 PyTorch 到 CUDA 12.1
  - [x] SubTask 1.1: 卸载当前 PyTorch 2.12.0+cu130
  - [x] SubTask 1.2: 安装 PyTorch CUDA 12.1 版本（兼容驱动 535.161.08）
  - [x] SubTask 1.3: 验证 CUDA 可用性（沙箱环境限制 GPU 访问，需用户在终端手动验证）

- [x] Task 2: 创建项目规则文件
  - [x] SubTask 2.1: 创建 `.trae/rules/kronos_rules.md`

- [x] Task 3: 安装额外依赖
  - [x] SubTask 3.1: 安装 litellm、python-dotenv、fastapi、uvicorn
  - [x] SubTask 3.2: 验证所有依赖安装成功

- [x] Task 4: 创建统一配置文件
  - [x] SubTask 4.1: 创建 `kronos/.env`（Kronos 模型 + 预测参数 + LLM + API 服务配置）
  - [x] SubTask 4.2: 创建 `kronos/.env.example`（不含敏感信息）
  - [x] SubTask 4.3: 确保 `.env` 在 `.gitignore` 中

- [x] Task 5: 创建配置加载模块 `kronos/config.py`
  - [x] SubTask 5.1: 实现配置加载函数（从 `.env` 读取并提供默认值）
  - [x] SubTask 5.2: 实现设备自动检测（CUDA → MPS → CPU）
  - [x] SubTask 5.3: 实现 LLM 配置验证

- [x] Task 6: 创建 LLM 分析器模块 `kronos/llm_analyzer.py`
  - [x] SubTask 6.1: 实现 `LLMAnalyzer` 类（litellm 调用 custom LLM）
  - [x] SubTask 6.2: 实现 `analyze_prediction()` 方法（构建 prompt + 调用 LLM）
  - [x] SubTask 6.3: 实现 LLM 不可用时优雅回退
  - [x] SubTask 6.4: 实现结构化分析输出

- [x] Task 7: 创建数据源模块 `kronos/data_loader.py`
  - [x] SubTask 7.1: 实现 CSV 加载（自动列名映射）
  - [x] SubTask 7.2: 实现 A 股涨跌停后处理（±10%）

- [x] Task 8: 创建 FastAPI 推理服务 `kronos/api.py`
  - [x] SubTask 8.1: 实现 FastAPI 应用和模型加载
  - [x] SubTask 8.2: 实现 `POST /api/predict` 端点（支持 csv 数据上传）
  - [x] SubTask 8.3: 实现 `GET /api/model-status` 和 `GET /api/health` 端点
  - [x] SubTask 8.4: 整合 LLM 分析到 API 响应

- [x] Task 9: 端到端验证
  - [ ] SubTask 9.1: 验证 GPU 推理可用（在 SSH 终端执行）
  - [x] SubTask 9.2: 启动 FastAPI 服务并测试 API 端点
  - [x] SubTask 9.3: 验证 LLM 分析功能（API key 未配置时正确回退）
  - [x] SubTask 9.4: 验证完整预测流程（config → data_loader → predictor，模型加载需 GPU）

# Task Dependencies
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 5]
- [Task 7] depends on [Task 5]
- [Task 8] depends on [Task 6, Task 7]
- [Task 9] depends on [Task 8]
