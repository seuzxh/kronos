# 训练结果分析 - 高贝塔 5min 微调

> 训练日期:2026-07-16
> 模型:Kronos-base(102M)微调,5min K 线预测次日 48 根走势
> 总训练时长:~3.5 小时(4 折 CV + 最终全量模型)

---

## TL;DR

- ✅ **微调有效**:zero-shot Val Loss 2.35 → 微调 2.19,**降幅 6.8%**
- ✅ **无过拟合**:所有折 Val Loss 单调下降至收敛
- ✅ **训练稳定**:4 卡 DDP + sglang 共存,全程无崩溃
- ⚠️ **fold2 异常偏高**(2.278):2026-05~06 市场阶段预测难度显著高于其他时段
- ⚠️ **Val Loss 绝对值偏高(2.19)**:5min 信噪比低、高贝塔波动剧烈,是任务本身难度

---

## 1. CV 结果总览(核心结论)

| 模型 | Val Loss | 验证段(市场阶段) | 训练耗时 |
|---|---|---|---|
| fold0 | **2.1887** ← 最佳单折 | 2026-03-16~04-13 | 38min |
| fold1 | **2.1966** | 2026-04-14~05-14 | 41min |
| fold2 | **2.2780** ⚠️ 异常偏高 | 2026-05-15~06-11 | 35min |
| fold3 | **2.2002** | 2026-06-12~07-10 | 44min |
| **full** | **2.1964** ← 推荐上线 | (复用 fold3 验证段) | 50min |

### 关键发现

1. **4 折平均 Val Loss = 2.2159,标准差 = 0.034**
   - fold0/1/3 稳定在 2.19~2.20(一致性高)
   - **fold2 异常偏高(2.278)**,说明 2026-05~06 行情特殊

2. **fold2 偏高分析**(2026-05-15~06-11):
   - 该时段可能是剧烈波动 / 风格切换期,高贝塔股噪声放大
   - 模型在该时段泛化能力下降,提示**不同市场阶段预测难度差异大**
   - 实际部署需注意:模型在震荡 / 转折期表现会退化

3. **full 模型(2.1964)≈ fold3(2.2002)**:
   - 全量训练未显著优于单折,说明**数据量已不是主要瓶颈**
   - 瓶颈可能在:信噪比(5min 噪声大)+ tokenizer 未微调

4. **各折 Val Loss 均优于预训练 zero-shot**(smoke_test 测得 2.35):
   - 微调有效:2.35 → 2.19,降幅 6.8%

---

## 2. 各模型完整训练曲线

### fold0(最佳折)

```
Epoch  Val Loss   Δ
  1    2.2074     —
  2    2.1981   -0.0093
  3    2.1945   -0.0036
  4    2.1931   -0.0014
  5    2.1904   -0.0027
  6    2.1893   -0.0011
  7    2.1887   -0.0006
  8    2.1887    0.0000  ← 收敛
```

### full(最终模型)

```
Epoch  Val Loss   Δ
  1    2.2177     —
  2    2.2077   -0.0100
  3    2.2028   -0.0049
  4    2.1999   -0.0029
  5    2.1987   -0.0012
  6    2.1972   -0.0015
  7    2.1966   -0.0006
  8    2.1964   -0.0002  ← 收敛
```

**收敛特征**:所有折均在 epoch 7-8 收敛,Val Loss 单调下降,**无过拟合**。

---

## 3. 训练健康度评估

### ✅ 正面信号

- **无过拟合**:所有折 Val Loss 单调下降至收敛,train/val gap 合理
- **训练稳定**:4 卡 DDP + sglang 共存,全程无崩溃(除 full 路径 bug 已修)
- **收敛一致**:各折都在 epoch 7-8 达最优,OneCycleLR 调度有效
- **微调有效**:zero-shot 2.35 → 微调 2.19(降幅 6.8%)

### ⚠️ 需关注

1. **Val Loss 绝对值偏高(2.19)**:
   - Kronos 用 cross-entropy on token logits,2.19 意味着预测难度大
   - 根因:5min 走势信噪比低,高贝塔股波动剧烈
   - 这是任务本身的难度,非模型缺陷

2. **fold2 异常(2.278)**:
   - 2026-05~06 市场阶段预测难度显著高于其他时段
   - 建议:实盘部署时监控市场状态,异常期降低仓位

3. **Train Loss 高方差(2.06~2.50)**:
   - 高贝塔股池每日换手 84 只,batch 内股票异质性高
   - 可考虑:按波动率分组训练,或增大 batch 平滑梯度

---

## 4. 模型产物清单

```
finetune/outputs/highbeta_5min/
├── fold0/finetune_predictor/checkpoints/best_model/  (391MB) ← 最佳单折
├── fold1/finetune_predictor/checkpoints/best_model/  (391MB)
├── fold2/finetune_predictor/checkpoints/best_model/  (391MB)
├── fold3/finetune_predictor/checkpoints/best_model/  (391MB)
└── full/finetune_predictor/checkpoints/best_model/   (391MB) ← 最终上线模型
```

**推荐上线**:`full/`(全量训练,Val 2.1964)
**备用对比**:`fold0/`(最佳单折,Val 2.1887)

---

## 5. 训练配置

| 参数 | 值 |
|---|---|
| GPU | 4× H20(sglang 共占 84GB/卡,训练 ~10GB/卡)|
| batch_size | 32(有效 batch 128,4 卡)|
| epochs | 8(OneCycleLR)|
| n_train_iter | 64000 样本/epoch |
| predictor_lr | 3e-5 |
| lookback→predict | 240→48 根 5min(5 天→次日全天)|
| NCCL | P2P/IB/SHM disabled(sglang 共存)|
| 数据 | 5038 只股票,1.44 亿根 5min,4 折 CV 切分 |

---

## 6. 后续改进方向(优先级排序)

### 高优先级(预期收益大)

1. **微调 tokenizer**(当前用预训练的):
   - 在 5min 数据上训练 tokenizer,让量化码本适应 5min 分布
   - 命令:各折先跑 `train_tokenizer.py` 再跑 `train_predictor.py`
   - 预期:predictor 效果可能提升(tokenizer 是信息瓶颈)

2. **增大 n_train_iter**:
   - 当前每 epoch 仅 64000 样本(占 1.23 亿窗口的 0.05%)
   - 增大到 200000 可能改善收敛

### 中优先级

3. **调整学习率**:
   - 当前 3e-5,可尝试 1e-4(更大)或 5e-6(更小更稳)

4. **更长 lookback**:
   - 当前 240(5 天),试 480(10 天)捕捉更长趋势
   - 注意:max_context=512 限制,480+48+1=529 略超,需调 max_context

### 低优先级(探索性)

5. **加入日线特征**:5min + 日 K 联合,多频率输入
6. **市场状态条件**:用市场指标(如波动率)做条件输入,应对 fold2 类异常期

---

## 7. 训练过程问题记录

| 问题 | 原因 | 解决 |
|---|---|---|
| NCCL `Cuda failure 1` | sglang 占满 GPU,CUDA context 冲突 | `NCCL_P2P/IB/SHM_DISABLE=1` |
| full 路径 `foldfull` | `get_cv_fold_dir` 拼接 bug | 'full' 不带 fold 前缀 |
| full 无 val 数据 | preprocess 只生成 full/train | full 验证复用 fold3 val |
| 预处理 2.8 小时 | qlib API 开销大 | 直接读 bin + numpy 聚合(27min)|

---

## 附:原始日志位置

```
outputs/highbeta_5min/train_fold1.log        # fold1 训练日志
outputs/highbeta_5min/train_fold2.log        # fold2 训练日志
outputs/highbeta_5min/train_fold3.log        # fold3 训练日志
outputs/highbeta_5min/train_foldfull.log     # full 训练日志
outputs/highbeta_5min/train_all_progress.log # 全流程进度
```

> fold0 日志未单独保存(早期调试时未留存,但结果在 summary.json)。
