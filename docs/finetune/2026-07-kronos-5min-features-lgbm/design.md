# 实验设计:Kronos 5min 特征提取器 + LGBM 选股集成

> **实验编号**:`2026-07-kronos-5min-features-lgbm`
> **状态**:进行中
> **日期**:2026-07-20
> **前置实验**:[实验 1 高贝塔 5min](../2026-07-highbeta-5min/README.md)、[实验 2 高贝塔指增](../2026-07-highbeta-enhancement/README.md)(均失败)

---

## 1. 背景:为什么换思路

前两次实验(端到端选股)都失败了:Val Loss 降 42% 但 RankIC ≈ 0。根因是 **Kronos 的架构-任务错配**:

```
选股需要: 横截面比较 100 只股票的相对强弱
Kronos 会: 自回归续写单条 K 线(没有横截面能力)
```

但 Kronos 学到的"K 线形态理解"能力本身是有价值的。**关键洞察**:别让 Kronos 端到端做选股(它不会),让它做它擅长的——**形态理解,产出特征**,再交给 LGBM 做横截面比较。

这是 Foundation Model 的标准用法:BERT/GPT 都不是端到端做下游任务,而是当特征提取器。

---

## 2. 核心设计

### 2.1 尺度对齐(关键修正)

本方案的核心是**与 LGBM 的 label horizon 严格对齐**:

| 维度 | 配置 | 与 label 的关系 |
|---|---|---|
| Kronos 输入 | 过去 5 交易日 5min(240 根) | 用 T 日开盘前可得信息 |
| Kronos 输出 | 预测 T 日全天 5min(48 根) | 覆盖 label 的前半段 |
| LGBM 决策 | T 日 9:41 买入 | 用 Kronos 的 T 日预测 |
| LGBM label | T+1 收盘卖出(≈1.5 天 horizon) | Kronos 预测与之对齐 |

**防泄露保证**:特征写 T 日,信息截止 T-1 收盘。T 日开盘前的所有决策都基于完整可得信息。

### 2.2 8 个特征定义

从 Kronos 预测的 48 根 5min K 线路径提取:

| 特征 | 含义 | 捕捉的信息 |
|---|---|---|
| `kronos_ret_intraday` | 预测 T 日累计收益(收/开-1) | 整体涨跌方向 |
| `kronos_high_ratio` | 日内最高涨幅 | 上行潜力 |
| `kronos_low_ratio` | 日内最大跌幅 | 下行风险 |
| `kronos_range` | 日内振幅 (high-low)/open | 波动性 |
| `kronos_close_pos` | 收盘相对位置 (收-低)/(高-低) | 强弱(收在高位 vs 低位) |
| `kronos_morning_ret` | 上午收益(13:00 vs 09:30) | 早盘动量 |
| `kronos_afternoon_ret` | 下午收益(15:00 vs 13:00) | 尾盘动量 |
| `kronos_uncertainty` | 5 次采样间收益标准差 | 模型信心度 |

**设计原则**:不只看"涨跌",而是从预测路径中提取多维信息(方向 + 波动 + 路径形态 + 信心度)。

### 2.3 多采样机制(关键技术细节)

Kronos 的自回归预测支持 `sample_count` 次并行采样。官方 `predict()` 方法在内部对多次采样取**平均**,丢失了路径间的差异。

本方案的 `predict_independent_paths` 直接调用 `auto_regressive_inference`,在 BSQ decode 后 **reshape 但不 mean**,拿到 `sample_count` 条独立路径。这样:
- 前 7 个特征:跨采样取平均(主特征)
- `kronos_uncertainty`:跨采样的标准差(信心度)

---

## 3. 数据流

```
1min bin(备份目录,2008~2026)
    ↓ _aggregate_1min_to_5min(位置分组,每日 48 根)
5min K 线(dict[symbol -> DataFrame])
    ↓ 逐日滚动:取 [T-5, T-1] 历史(240 根)
Kronos 推理(sample_count=5 并行)
    ↓ predict_independent_paths_batch
5 条独立预测路径 (5, 48, 6)
    ↓ extract_features
8 个标量特征
    ↓ write_stock_features(对齐 close.day.bin)
qlib_root/features/<code>/kronos_*.day.bin
    ↓ MinuteEnhancedKronosHandler.get_feature_config
LGBM 特征矩阵(26 列 = 18 baseline + 8 kronos)
```

---

## 4. Ablation 设计

三跑对比,严格 OOS:

| 跑 | Handler | 特征数 | Kronos 特征来源 |
|---|---|---|---|
| **A** | `MinuteEnhancedHandler` | 18 | 无(baseline) |
| **B** | `MinuteEnhancedKronosHandler` | 26 | Kronos-base(预训练 zero-shot) |
| **C** | `MinuteEnhancedKronosHandler` | 26 | Kronos-finetune(highbeta_5min 微调版) |

**时间切分**(三跑一致):
- train: 2024-01-01 → 2025-12-31
- valid: 2026-01-01 → 2026-03-31
- **test: 2026-04-01 → 2026-07-02**(OOS,不参与任何选择)

**对照线**:champion baseline(18 因子)OOS IC 0.059 / RankIC 0.073 / 年化超额 51%。

---

## 5. 执行顺序

因为 B/C 共用同一组 bin 文件名,必须串行执行:

```
1. 提取 base 版特征(覆盖 bin)       ~5h
2. 跑 LGBM-B                          ~30min
3. 备份 B 结果
4. 提取 finetune 版特征(覆盖 bin)    ~5h
5. 跑 LGBM-C                          ~30min
6. 跑 LGBM-A(任何时候,不依赖 bin)    ~30min
7. 三跑对比 + 报告                    ~1h
```

---

## 6. 成功/失败判定

| 结果 | 判定 | 后续动作 |
|---|---|---|
| B 或 C 的 OOS IC ≥ baseline + 5% | ✅ 成功 | Kronos 特征有价值,继续优化 |
| 训练段 IC 升但 test 段不升 | ⚠️ 过拟合 | 记下教训,微调版在训练股池过拟合 |
| B/C 都不升,或 test 段下降 | ❌ 失败 | 第三次诚实归档:Kronos 对 A 股选股无增量 |

**输不了**:最差结果给"Kronos 不适合 A 股选股"加上第三次铁证。

---

## 7. 代码位置

| 组件 | 路径 |
|---|---|
| 特征提取脚本 | `finetune/feature_extraction/extract_features.py` |
| 特征定义 | `finetune/feature_extraction/feature_defs.py` |
| 配置 | `finetune/feature_extraction/config_5min_feat.py` |
| 后台运行 | `finetune/feature_extraction/run_extract.sh` |
| LGBM 集成(handler) | `3.qlib_ifind_beta/qlib_ifind_beta/minute_enhanced_handler.py` |
| LGBM 集成(字段) | `3.qlib_ifind_beta/qlib_ifind_beta/config.py:KRONOS_FIELDS` |
| Ablation yaml | `3.qlib_ifind_beta/qrun/workflow_kronos_ablation_{A,B,C}.yaml` |
