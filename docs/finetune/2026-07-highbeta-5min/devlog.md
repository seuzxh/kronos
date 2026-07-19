# 开发日志 - 高贝塔 5min 微调

> 本文档记录实施过程中的**技术细节、踩坑与解决方案**,便于 code review 和后续维护。
> 决策层面的内容见 [design.md](design.md)。

---

## TL;DR(主要踩坑速览)

| 坑 | 影响 | 解决 |
|---|---|---|
| qlib 1min 单日查询返回空 | 拉不到数据 | end_date + 1 天 |
| `resample('5min')` 产生 50 根而非 48 | 数据错位 | 按位置每 5 根分组 |
| vwap 口径与 close 不一致 | amount 错算 | 用 `amt = close * volume` |
| qlib API 拉数据 2.8 小时 | 不可接受 | 直接读 bin + numpy 聚合(27 分钟)|
| pickle DataFrame index 列名 | Dataset 崩溃 | 重命名首列为 'datetime' |
| NCCL Cuda failure 1 | sglang 共存冲突 | 禁用 P2P/IB/SHM |
| full 路径拼接 `foldfull` | 找不到数据 | 'full' 不带 fold 前缀 |

---

## 1. qlib 1min 查询的坑

**现象**:`D.features(freq='1min', start='2026-07-10', end='2026-07-10')` → **返回空!**

**原因**:1min 频率下,单日边界 end 必须跨到次日。

**正确写法**:
- 日期范围:`start='2026-06-28', end='2026-07-04'`
- 或带时间戳

**修复**:`preprocess_5min.py` 中 end_date + 1 天。

---

## 2. 1min → 5min 聚合的时间对齐坑

**A 股 1min 时间戳**:9:31(首)~ 15:00(末),共 240 根/天。

**❌ 错误做法**:`resample('5min').agg(...)`
- 会产生 **50 根**(9:30 bin 只含 4 根,15:00 bin 只含 1 根)
- 首根 volume 不对

**✅ 正确做法**:**基于位置分组**(每 5 根聚 1 根)
- 时间戳取每组最后一根 1min 时间
- 函数:`_aggregate_1min_to_5min()`
- 逻辑:按日分组 → 每日内按位置每 5 根 → 时间戳=组内最后 1min

**聚合规则**:
```
open  = first
high  = max
low   = min
close = last
vol   = sum
amt   = sum
```

---

## 3. vwap 口径问题

qlib 1min bin 数据:
- `close` 是**后复权**(如 95.76)
- `vwap` 是**不复权**(如 8.97)
- `factor` ≈ 10.66

**❌ 不能用** `amt = vwap * volume`(口径混乱)

**✅ 用** `amt = close * volume`(后复权口径量能)

Kronos 做窗口 z-score 归一化,只用相对量能信号,绝对值不影响。

---

## 4. 性能优化(关键!)

### 问题
首版用 qlib API(`D.features`)拉取:100 只/批 × 196s,5114 只需 **2.8 小时**,不可接受。

### 优化 1:直接读 bin 文件(绕过 qlib API)

- qlib bin 格式:`np.frombuffer`,arr[0]=start_idx,arr[1:]=float32 数据
- 5 只 × 5 字段读取仅 5.7ms(vs qlib API 秒级)
- 数据完全一致(SH600000 2026-07-10 close=95.761 两方式相同)
- 函数:`_read_bin()`, `fetch_5min_bin_direct()`

### 优化 2:numpy 向量化聚合(替代 pandas groupby)

- 原 `_aggregate_1min_to_5min` 用 `groupby('_grp').agg()` + `apply(lambda)` → 慢
- 新版:`vals.values.reshape(n5, 5)` 后 axis=1 numpy 聚合(max/min/sum 向量化)
- 10 只:19s → 3.2s(快 6 倍)
- 5114 只预估:**27 分钟**(vs 优化前 2.8 小时)

**最终方案**:直接读 bin + numpy 聚合,5114 只约 27 分钟。

---

## 5. CV 切分边界验证(防未来泄露)

扩张式切分,实际验证:

```
折叠0: 训练[~2026-03-13] | 验证[2026-03-16~2026-04-13]  (训练终点 < 验证起点 ✅)
折叠1: 训练[~2026-04-13] | 验证[2026-04-14~2026-05-14]
折叠2: 训练[~2026-05-14] | 验证[2026-05-15~2026-06-11]
折叠3: 训练[~2026-06-11] | 验证[2026-06-12~2026-07-10]  ← 最近行情
最终:  训练[全量 2024-01-02~2026-07-10]
```

---

## 6. Dataset 关键改动

`dataset.py` 新增 `HighbetaDataset` 类,复用 `QlibDataset` 归一化逻辑。

**关键 bug 修复**:pickle 中 DataFrame 的 index 无名,`reset_index()` 后列名是 'index' 非 'datetime'。

```python
# 修复
if 'datetime' not in df.columns:
    df.rename(columns={df.columns[0]: 'datetime'})
```

### Dataset 验证结果(fold0/train,10 只股票测试)

```
候选窗口数:249,410
每 epoch 采样:32,000
Sample feature shape:torch.Size([289, 6])  ✅ (240+48+1)
Sample time shape:torch.Size([289, 5])     ✅
Feature range:[-1.690, 5.000] (clip=5.0 生效) ✅
```

---

## 7. 训练脚本改动

`train_predictor.py` / `train_tokenizer.py`:
- 环境变量 `KRONOS_CONFIG` / `KRONOS_FOLD` 切换配置
- `comet_ml` 改条件导入(`use_comet=False` 时不强制安装)
- 保存路径含 fold:`outputs/highbeta_5min/fold{N}/finetune_predictor/`
- 高贝塔模式 tokenizer 用预训练的(不单独训练)

---

## 8. check_data 验证结果(fold0)

```
[fold0/train] 股票数:5015, 总 124,453,729 根, 可用窗口 123,009,409
  OHLC 一致性违规:0 ✅
  时间范围:2024-01-02 ~ 2026-03-13
[fold0/val] 股票数:5022, 可用窗口 3,363,072
  验证段全部 48 根/天 ✅
```

### smoke_test 验证结果 ✅

```
HighbetaDataset:1.23 亿候选窗口, 每 epoch 采样 32000
Tokenizer 加载:d_in=6 ✅
前向:(1,289,6) → tokenize(1,289) → logits(1,288,1024) ✅
Loss:2.3481 (s1: 2.7430, s2: 1.9533) ✅
所有维度正确, 训练管线就绪
```

---

## 9. 数据异常发现

2024 年初部分交易日 5min 根数 ≠ 48:

| 日期 | 实际根数 | 应为 |
|---|---|---|
| 2024-01-03 | 12 | 48 |
| 2024-01-12 | 17 | 48 |

**原因推测**:2024 年初 1min 数据不完整(半天或数据缺失)。

**影响**:极小。`dataset.py` 的窗口切片要求 window=289,这些短日会被自然跳过(`num_samples = series_len - window + 1`,短日贡献 0 或极少窗口)。**无需处理**,仅记录。

---

## 10. 文件清单(本分支新增/修改)

| 文件 | 类型 | 说明 |
|---|---|---|
| `docs/` | 新增 | 本文档目录 |
| `finetune/config_highbeta.py` | 新增 | 高贝塔 5min 配置 |
| `finetune/universe.py` | 新增 | 股池加载 |
| `finetune/preprocess_5min.py` | 新增 | 1min→5min 聚合 + CV 切分 |
| `finetune/check_data.py` | 新增 | 数据验证脚本 |
| `finetune/dataset.py` | 修改 | 支持 config/fold 切换 |
| `finetune/train_predictor.py` | 修改 | 环境变量切换配置 |
| `finetune/train_tokenizer.py` | 修改 | 环境变量切换配置 |
| `finetune/run_cv.sh` | 新增 | CV 训练流水线 |
| `finetune/backtest_5min.py` | 新增 | 5min 回测脚本 |
| `finetune/summarize_backtest.py` | 新增 | 4 折回测汇总 |

---

## 11. Review 检查点(代码审查重点)

1. **CV 切分边界**:`preprocess_5min.py` 中 4 折的时间戳计算是否正确(无未来泄露)
2. **5min 聚合正确性**:open=first / high=max / low=min / close=last / volume=sum
3. **跨日连续性**:pickle 中每股票序列是否保留了跨日(隔夜缺口),没有被错误截断
4. **归一化**:`dataset.py` 仍只用 lookback 段算 mean/std(防泄露)
5. **样本 shape**:应为 (289, 6) + (289, 5),289 = 240 + 48 + 1
