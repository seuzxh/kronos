# Tasks

- [x] Task 1: 修改 data_loader.py — volume 必填 + 新增 load_qlib()
  - [x] SubTask 1.1: 将 `REQUIRED_COLUMNS` 改为 `["open", "high", "low", "close", "volume"]`
  - [x] SubTask 1.2: 删除 `load_csv()` 中 volume 缺失时填充 0.0 的逻辑
  - [x] SubTask 1.3: 新增 `load_qlib(symbol, start_time, end_time, provider_uri, freq)` 函数
  - [x] SubTask 1.4: qlib 不可用时（未安装/目录不存在）给出明确提示

- [x] Task 2: 更新 config.py 和 .env — 新增 qlib 配置 + 批量限制
  - [x] SubTask 2.1: KronosConfig 新增 `qlib_provider_uri` 字段
  - [x] SubTask 2.2: KronosConfig 新增 `max_batch_size` 字段（默认 50）
  - [x] SubTask 2.3: `.env` 和 `.env.example` 新增 `QLIB_PROVIDER_URI` 和 `MAX_BATCH_SIZE` 配置

- [x] Task 3: 更新 api.py — 新增 predict-qlib 端点
  - [x] SubTask 3.1: 新增 `POST /api/predict-qlib` 端点（symbol, start_time, end_time 参数）
  - [x] SubTask 3.2: qlib 未安装时返回 503 提示

- [x] Task 4: 安装 pyqlib 依赖
  - [x] SubTask 4.1: 在 kronos conda 环境中安装 pyqlib
  - [x] SubTask 4.2: 验证 qlib 数据可正常读取

- [x] Task 5: 更新文档
  - [x] SubTask 5.1: 更新 `kronos_rules.md` 数据格式要求章节（volume 必填 + qlib 数据源）
  - [x] SubTask 5.2: 更新 `design.md` 输入数据要求和数据获取方式章节

- [ ] Task 6: 验证并提交
  - [ ] SubTask 6.1: 验证不含 volume 的 CSV 加载时抛出 ValueError
  - [ ] SubTask 6.2: 验证含 volume 的 CSV 正常加载
  - [ ] SubTask 6.3: 验证 load_qlib() 可正常读取数据
  - [ ] SubTask 6.4: 验证 /api/predict-qlib 端点
  - [ ] SubTask 6.5: Git 提交

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1, Task 2]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 1, Task 2, Task 3]
- [Task 6] depends on [Task 1, Task 2, Task 3, Task 4, Task 5]
