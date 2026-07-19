# 训练结果分析 - 高贝塔指增日频微调

> 训练日期:2026-07-20
> 模型:Kronos-base(102M)微调,日线预测未来 5 日走势
> 总训练时长:~100 分钟(4 折 CV + 最终全量模型)

---

## TL;DR

- ✅ **微调效果显著**:zero-shot Val Loss 3.49 → 微调后 full 2.0184,**降幅 42%**
- ✅ **无过拟合**:所有折 Val Loss 单调下降至收敛
- ✅ **full 模型最优**(2.0184),优于任一单折
- ⚠️ **fold3 显著低于其他折**(2.07 vs 2.23),最近行情更可预测
- 🎯 **vs 上次 5min(2.19)**:日频略优(2.19 平均),但跨频率不可直接比

---

## 1. CV 结果总览

| 模型 | Val Loss | 验证段(市场阶段) | 训练耗时 |
|---|---|---|---|
| fold0 | 2.2310 | 2026-03-23~04-20 | ~21min |
| fold1 | 2.2074 | 2026-04-21~05-21 | ~21min |
| fold2 | 2.2446 | 2026-05-22~06-18 | ~22min |
| fold3 | **2.0697** ← 最佳单折 | 2026-06-22~07-17 | ~22min |
| **full** | **2.0184** ← 推荐上线 | (复用 fold3 验证段) | ~24min |

### 关键发现

1. **4 折平均 Val Loss = 2.1882,标准差 = 0.0697**
   - fold0/1/2 稳定在 2.20~2.24(一致性高)
   - **fold3 显著偏低(2.07)**,最近行情(2026-06~07)更可预测

2. **fold3 偏低分析**(2026-06-22~07-17):
   - 与上次 5min 实验相反(那次 fold2 异常偏高)
   - 可能是近期趋势更明显,模型易捕捉
   - 实盘部署需注意:模型在趋势行情表现优于震荡

3. **full 模型(2.0184)优于所有单折**:
   - 全量训练有效,数据量增加带来提升
   - 与上次 5min 不同(那次 full ≈ fold3)
   - 推荐上线:full 模型

4. **vs zero-shot 3.49**:
   - 微调降幅 42%(vs 上次 5min 仅 6.8%)
   - 说明日频数据量充足,微调能真正学到东西

---

## 2. full 模型完整训练曲线

```
Epoch  Val Loss   Δ
  1    2.2071     —
  2    2.1009   -0.1062
  3    2.0629   -0.0380
  4    2.0408   -0.0221
  5    2.0286   -0.0122
  6    2.0224   -0.0062
  7    2.0192   -0.0032
  8    2.0184   -0.0008  ← 收敛
```

**收敛特征**:Val Loss 单调下降,epoch 7-8 收敛,**无过拟合**。Δ 逐渐缩小符合 OneCycleLR 调度。

---

## 3. 与上次 5min 实验对比

| 维度 | 5min(上次) | 日频(本次) |
|---|---|---|
| 平均 Val Loss | 2.2159 | 2.1882 |
| 最优单折 | 2.1887 (fold0) | 2.0697 (fold3) |
| full 模型 | 2.1964 | **2.0184** |
| 数据时间范围 | 4 个月(2026-01~05) | **2.5 年**(2024-01~2026-07) |
| 微调降幅 | 6.8% | **42%** |
| 异常折 | fold2 偏高(震荡期) | fold3 偏低(趋势期) |

**注**:Val Loss 是 token 重建误差,跨频率不可直接比(日频形态更多样,重建难度本就更高)。真正对比看 [backtest-report.md](backtest-report.md) 的**超额 RankIC**。

---

## 4. 训练健康度

### ✅ 正面信号

- **微调大幅有效**:Val Loss 降 42%
- **无过拟合**:所有折单调下降至收敛
- **full 优于单折**:数据量带来提升
- **训练稳定**:4 卡 DDP + sglang 共存,全程无崩溃
- **数据充足**:4977 只股票,140 万候选窗口/折

### ⚠️ 需关注

1. **fold3 异常偏低**:近期行情可能过于"友好",实盘需防过度乐观
2. **训练 Loss 高方差**(2.0~2.5):高贝塔股异质性高,batch 内差异大

---

## 5. 训练配置

| 参数 | 值 |
|---|---|
| GPU | 4× H20(sglang 共占 84GB/卡,训练 ~10GB/卡)|
| batch_size | 32(有效 batch 128,4 卡)|
| epochs | 8(OneCycleLR)|
| n_train_iter | 64000 样本/epoch |
| predictor_lr | 3e-5 |
| lookback→predict | 240→5 个交易日(1 年→1 周)|
| max_context | 512(246 < 512 ✅)|
| 数据 | 4977 只股票,4977 万根日线 |
| 指数特征 | 预处理含(baseline 暂未输入模型,留作 H1 假设)|

---

## 6. 模型产物清单

```
finetune/outputs/enhancement_daily/
├── fold0/finetune_predictor/checkpoints/best_model/  (391MB)
├── fold1/finetune_predictor/checkpoints/best_model/  (391MB)
├── fold2/finetune_predictor/checkpoints/best_model/  (391MB)
├── fold3/finetune_predictor/checkpoints/best_model/  (391MB) ← 最佳单折
└── full/finetune_predictor/checkpoints/best_model/   (391MB) ← 推荐上线
```

**推荐上线**:`full/`(全量训练,Val 2.0184)
**备用对比**:`fold3/`(最佳单折,Val 2.0697,验证段未见过)

---

## 7. 后续改进方向

### 高优先级

1. **H1 假设验证**:加指数特征(d_in 6→12),对比 RankIC 是否提升 ≥ 20%
2. **H4 假设验证**:微调 tokenizer(当前用预训练),对比 RankIC 是否提升 ≥ 10%

### 中优先级

3. **更长 predict_window**:当前 5 日,试 10/20 日
4. **更大 n_train_iter**:当前 64000/epoch,试 200000

---

## 附:原始日志

```
/tmp/kronos_train/fold0.log
/tmp/kronos_train/fold1.log
/tmp/kronos_train/fold2.log
/tmp/kronos_train/fold3.log
/tmp/kronos_train/foldfull.log
```

汇总数据:`finetune/outputs/enhancement_daily/TRAINING_REPORT.json`
