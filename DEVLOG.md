# 高贝塔股池 5min 微调 - 开发日志

> 本文件记录所有实施细节,便于 review。按时间倒序,最新在上。

---

## 2026-07-16 实施记录

### 项目背景
- **目标**: 微调 Kronos-base,预测高贝塔股池(883926.TI)成分股**次日全天48根5min走势**
- **分支**: `feat/highbeta-5min-finetune` (从 master `67b630e` 切出)

### 关键决策(已与用户确认)
| 决策点 | 选择 | 理由 |
|---|---|---|
| 预测频率 | 5min K线(由1min聚合) | 用户指定 |
| 股池范围 | 严格用历史股池 universe_snapshots.csv | 贴合实际部署 |
| 训练窗口 | lookback=240(5交易日)→predict=48(次日全天),允许跨日 | 训练/推理结构一致 |
| 时间切分 | **4折扩张式时间序列CV + 最终全量模型** | 数据仅2.5年,不做每日滚动 |
| 代码路径 | 基于 `finetune/` (qlib路径)扩展 | 原生支持多股票DDP,finetune_csv只支持单股票 |

### 数据源
- **1min K线**: `/home/zxh/cn_data_1min` (qlib bin格式,2024-01-02起,5540只,240根/天)
- **日K(日历用)**: `/home/zxh/qlib_data`
- **股池**: `/home/zxh/projects/3.qlib_ifind_beta/data/universe_snapshots.csv` (609日×100只)
- **黑名单**: SH600599/SH600636/SH600696/SH688287 (1min factor退化)

### CV 切分设计(4折×验证20日)
```
交易日轴  2024-01-02(第1日) ─────────────────── 2026-07-10(第609日)

折叠0: [训练469日 ████████][验证20日▓▓]          val=[第470-489日]
折叠1: [训练489日 ████████████][验证20日▓▓]      val=[第490-509日]
折叠2: [训练509日 ████████████████][验证20日▓▓]  val=[第510-529日]
折叠3: [训练529日 ████████████████████][验证20日▓▓] val=[第590-609日]
                                                          ↑最近行情
最终:  [训练609日 全量 ████████████████████████████████] → 上线
```
- 注: 4个验证段共80日,中间无间隔(连续滚动),训练段扩张式递增
- 每折 pickle 路径: `finetune/data/fold{0-3}/{train,val}_5min.pkl`

### 实施步骤完成情况
- [x] Step1: 分支 + config_highbeta.py ✅
- [x] Step2: universe.py 股池加载 ✅
- [x] Step3: preprocess_5min.py + check_data.py (全量预处理后台运行中) ✅
- [x] Step4: dataset.py 适配 + 训练脚本环境变量 ✅
- [x] Step5: run_cv.sh 流水线 ✅
- [ ] Step6: 端到端验证 (Dataset已验证通过,待全量数据跑通)

### ⚠️ 重要发现 (Step2)
**股池并集 = 5114 只股票**(远超方案预估的500-800只)!
- 原因: 高贝塔股池每日100只但**换手率极高**,相邻日重叠仅5只(新进95/退出95)
- 即"高贝塔"是个动态属性,几乎所有A股都曾在某个时点进入过高贝塔池
- **影响**: preprocess 不能无脑拉5114只全量1min数据(数据量爆炸)
- **应对**: preprocess 按CV fold的时段,只拉该时段股池并集(每折可能仍上千只,需分批拉取)
- 这个换手率也解释了为什么 universe_snapshots.csv 有609日×100=6万行,但并集5114只接近全市场

### Step2 验证结果
```
日期数: 609 (2024-01-02 → 2026-07-10)
每日股票数: min=98 max=100 mean=99.9
历史并集股票数: 5114
换手率检查 (相邻日): 重叠5只, 新进95只, 退出95只
黑名单4只: 全部✅已排除
```

### Step3 关键技术细节 (Review 重点!)

**1. qlib 1min 查询的坑**:
- `D.features(freq='1min', start='2026-07-10', end='2026-07-10')` → **返回空!**
- 原因: 1min频率下,单日边界 end 必须跨到次日
- 正确: `start='2026-06-28', end='2026-07-04'` (日期范围) 或带时间戳
- preprocess_5min.py 已处理: end_date + 1天

**2. 1min→5min聚合的时间对齐坑**:
- A股1min时间戳: 9:31(首)~15:00(末),共240根/天
- 用 `resample('5min')` 会产生 **50根** (9:30bin只含4根, 15:00bin只含1根)
- ❌ 错误: `resample('5min').agg(...)` → 50根,首根volume不对
- ✅ 正确: **基于位置分组** (每5根聚1根), 时间戳取每组最后一根1min时间
- `_aggregate_1min_to_5min()` 函数: 按日分组 → 每日内按位置每5根 → 时间戳=组内最后1min

**3. vwap 口径问题**:
- qlib 1min bin: close 是后复权(95.76), vwap 是**不复权**(8.97), factor≈10.66
- ❌ 不能用 `amt = vwap * volume` (口径混乱)
- ✅ 用 `amt = close * volume` (后复权口径量能)
- Kronos 做窗口z-score归一化,只用相对量能信号,绝对值不影响

**4. CV切分边界验证** (扩张式, 无未来泄露):
```
折叠0: 训练[~2026-03-13] | 验证[2026-03-16~2026-04-13]  (训练终点 < 验证起点 ✅)
折叠1: 训练[~2026-04-13] | 验证[2026-04-14~2026-05-14]
折叠2: 训练[~2026-05-14] | 验证[2026-05-15~2026-06-11]
折叠3: 训练[~2026-06-11] | 验证[2026-06-12~2026-07-10]  ← 最近行情
最终:  训练[全量2024-01-02~2026-07-10]
```

### Step4 关键改动
- `dataset.py`: 新增 `HighbetaDataset` 类, 复用QlibDataset归一化逻辑
  - 关键bug修复: pickle中DataFrame的index无名, `reset_index()`后列名是'index'非'datetime'
  - 修复: `if 'datetime' not in df.columns: df.rename(columns={df.columns[0]:'datetime'})`
- `train_predictor.py` / `train_tokenizer.py`: 环境变量 `KRONOS_CONFIG`/`KRONOS_FOLD` 切换
  - comet_ml 改条件导入 (use_comet=False时不强制安装)
  - 保存路径含 fold: `outputs/highbeta_5min/fold{N}/finetune_predictor/`
  - 高贝塔模式 tokenizer 用预训练的(不单独训练)

### Step4 验证结果 (HighbetaDataset)
```
加载 fold0/train (10只股票测试数据):
  候选窗口数: 249,410
  每epoch采样: 32,000
  Sample feature shape: torch.Size([289, 6])  ✅ (240+48+1)
  Sample time shape: torch.Size([289, 5])     ✅
  Feature range: [-1.690, 5.000] (clip=5.0生效) ✅
```

### 全量预处理估算
- 5114只股票, 52批(100只/批), 预计~35分钟
- 产物: 4折 × {train~7GB, val~0.3GB} + full~9GB, 共约30GB
- 后台进程运行中 (PID 3625305)

### ⚠️ 性能优化记录 (Step3 重大调整)

**问题**: 首版用 qlib API (`D.features`) 拉取, 100只/批×196s, 5114只需 **2.8小时**, 不可接受。

**优化1: 直接读bin文件 (绕过qlib API)**
- qlib bin格式: `np.frombuffer`, arr[0]=start_idx, arr[1:]=float32数据
- 5只×5字段读取仅5.7ms (vs qlib API秒级)
- 数据完全一致 (SH600000 2026-07-10 close=95.761 两方式相同)
- 函数: `_read_bin()`, `fetch_5min_bin_direct()`

**优化2: numpy向量化聚合 (替代pandas groupby)**
- 原 `_aggregate_1min_to_5min` 用 `groupby('_grp').agg()` + `apply(lambda)` → 慢
- 新版: `vals.values.reshape(n5,5)` 后 axis=1 numpy聚合 (max/min/sum向量化)
- 10只: 19s → 3.2s (快6倍)
- 5114只预估: **27分钟** (vs 优化前2.8小时)

**最终方案**: 直接读bin + numpy聚合, 5114只约27分钟, 后台运行中。

### 文件清单(本分支新增/修改)
| 文件 | 类型 | 说明 |
|---|---|---|
| `DEVLOG.md` | 新增 | 本文件 |
| `finetune/config_highbeta.py` | 新增 | 高贝塔5min配置 |
| `finetune/universe.py` | 新增 | 股池加载 |
| `finetune/preprocess_5min.py` | 新增 | 1min→5min聚合+CV切分 |
| `finetune/check_data.py` | 新增 | 数据验证脚本 |
| `finetune/dataset.py` | 修改 | 支持 config/fold 切换 |
| `finetune/train_predictor.py` | 修改 | 环境变量切换配置 |
| `finetune/train_tokenizer.py` | 修改 | 环境变量切换配置 |
| `finetune/run_cv.sh` | 新增 | CV训练流水线 |

### Review 检查点
明天 review 时重点关注:
1. **CV切分边界**: `preprocess_5min.py` 中4折的时间戳计算是否正确(无未来泄露)
2. **5min聚合正确性**: open=first/high=max/low=min/close=last/volume=sum
3. **跨日连续性**: pickle中每股票序列是否保留了跨日(隔夜缺口),没有被错误截断
4. **归一化**: dataset.py 仍只用lookback段算mean/std(防泄露)
5. **样本shape**: 应为 (289,6)+(289,5),289=240+48+1

---

## 明天如何继续 (操作指引)

### 1. 确认预处理产物
```bash
cd /home/zxh/projects/Kronos/finetune
ls data/highbeta_5min/        # 应有 fold0-3, full 目录
# 验证数据
python check_data.py --fold 0
python check_data.py --fold full
```

### 2. 跑 smoke test (验证训练管线, ~1分钟, 单GPU)
```bash
KRONOS_FOLD=0 python smoke_test.py
# 期望输出: ✅ Smoke test 通过
```

### 3. 启动4折CV训练 (~每折2-4小时, 4卡)
```bash
# 单独训练某折
bash run_cv.sh train 0
# 或顺序训练全部4折
bash run_cv.sh train-all
```

### 4. 检查各折 val_loss, 选最优配置
```bash
# 各折 summary.json 在 outputs/highbeta_5min/fold{N}/finetune_predictor/
cat outputs/highbeta_5min/fold0/finetune_predictor/summary.json
```

### 5. 训练最终全量模型
```bash
bash run_cv.sh train-final
```

### 当前状态 (2026-07-16 02:40)
- 代码: 全部完成并提交 (commit on feat/highbeta-5min-finetune)
- 数据: 全量预处理后台运行中 (5114只, 约27分钟, 预计02:50完成)
- 待办: 预处理完成后跑 check_data + smoke_test 验证
