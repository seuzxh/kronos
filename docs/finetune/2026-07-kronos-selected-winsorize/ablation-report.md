# Ablation 报告:实验 4 Kronos 精选 + Winsorize

> **状态**:✅ 完成(判定性结论)

## 1. 四跑对比表

| 跑 | 特征数 | 处理 | IC | RankIC | vs A baseline |
|---|:---:|:---:|:---:|:---:|:---:|
| **A baseline** | 18 | 无 | 0.0548 | 0.0612 | — |
| B +base | 26 | 无 | 0.0329 | 0.0439 | ❌ base 看跌偏差 |
| C +全 finetune | 26 | 无 | 0.0491 | 0.0651 | RankIC +6.4% |
| **D +精选+Winsorize** | 21 | 3 字段截尾 | **0.0553** | **0.0656** | RankIC **+7.2%** |

**门槛**(RankIC ≥ 0.0612 × 1.05 = 0.0643):D 跑 0.0656 **达标** ✅

## 2. 核心判定

### 2.1 D vs C:提升微弱(RankIC +0.0005)

```
C 全 Kronos:    RankIC = 0.0651
D 精选+Wins:    RankIC = 0.0656  (+0.0005,+0.8%)
```

精选 3 个强特征 vs 全量 8 个,RankIC 几乎无差。这证明:**LGBM 对冗余/噪声特征有鲁棒性** —— 即使喂 5 个噪声特征,LGBM 也会自动给低权重,不会显著拖累排序。

### 2.2 D 的 IC 恢复到 baseline(Winsorize 生效)

```
A baseline:     IC = 0.0548
C 全 Kronos:    IC = 0.0491  (-10%,异常值扰动)
D 精选+Wins:    IC = 0.0553  (+0.9%,恢复到 baseline)
```

**Winsorize 确实去除了异常值**,IC 从 0.0491 恢复到 0.0553(甚至略超 baseline)。这说明 C 跑 IC 下降的根因是 Kronos 特征的极端值,截尾后问题解决。

### 2.3 Kronos 特征天花板:RankIC +7%

D 是理论上"最干净的 Kronos 用法":3 个最强特征 + 去异常值。结果 RankIC +7.2%,与 C(+6.4%)相差无几。

**结论:Kronos 特征在当前框架下的天花板约 RankIC +7%**。精选和 Winsorize 能让 IC 更干净,但不会突破 RankIC 的上限。

## 3. 为什么 RankIC 天花板是 +7%?

回到单特征诊断,Kronos 最强的 `range` 在 T+1 的 RankIC 只有 -0.054。即使完美集成,单个弱信号能贡献的边际 RankIC 就是几个百分点。

关键发现 —— **horizon 错配**:

```
kronos_range 的 RankIC 随 horizon:
  T+1:  -0.054  ← LGBM 现在用的
  T+5:  -0.135  ← 2.5 倍
  T+10: -0.160  ← 3 倍
```

Kronos 的反转信号是**中期**的,但 LGBM 的 label 是 T+1(短期)。**强信号被弱 label 稀释了**。这是 RankIC 天花板的根本原因。

## 4. 两个假设的验证

| 假设 | 结果 | 说明 |
|---|---|---|
| 删噪声特征后 RankIC 提升 | ❌ 不成立 | LGBM 对冗余鲁棒,8→3 几乎无差 |
| Winsorize 后 IC 恢复 | ✅ 成立 | IC 从 0.0491 恢复到 0.0553 |

**假设 1 失败的根因**:树模型(LGBM)对特征冗余天然鲁棒。8 个特征里有 5 个噪声,LGBM 通过信息增益自动过滤,RankIC 不受影响。这与线性模型不同(线性模型会被冗余特征的多重共线性干扰)。

**假设 2 成功的意义**:Winsorize 修复了 IC,虽然不提升 RankIC,但让模型更稳健(分裂点不被极端值带偏)。这是个**工程改善**,不是 alpha 提升。

## 5. 后续方向(重要)

本实验是**判定性节点**:当前框架(T+1 label + 5min Kronos + LGBM)下,Kronos 特征天花板约 RankIC +7%。

要突破,必须换框架。按预期增益排序:

### 方向 1:中期反转策略(最推荐,预期 RankIC +15~20%)

把 label 从 T+1 改成 T+5 或 T+10,对齐 Kronos 的强信号 horizon。需要:
- 改 handler 的 label 表达式(`Ref($close,-5)/$price_941 - 1`)
- 改策略的持仓周期(hold_thresh=5)
- 可能要重新调 topk/n_drop

理论依据:`range` 在 T+10 的 RankIC 是 T+1 的 3 倍,如果 label 对齐,整体 RankIC 可能从 +7% 提升到 +15% 以上。

### 方向 2:多尺度特征融合(预期 RankIC +3~5%)

加日线 Kronos 特征(已有 `enhancement_daily` checkpoint),捕捉不同周期的反转:
- 5min Kronos:短期形态(现在用的)
- 日线 Kronos:中期趋势
- 两套特征叠加,互补不同周期

### 方向 3:特征交互(预期 RankIC +2~3%)

构造 Kronos 反转 × 现有动量的交互项:
- `kronos_range × startup_mom_5m`(动量+反转的张力)
- `kronos_uncertainty × vol_vs_yest`(不确定性+量能)

捕捉"动量到反转的拐点"。

## 6. 最终结论

**Kronos 特征精选 + Winsorize 的价值**:
- RankIC:+7.2%(与全量 +6.4% 相比,提升微弱)
- IC:恢复到 baseline(去掉异常值扰动)
- 工程鲁棒性:提升(分裂点更稳)

**判定**:当前框架下 Kronos 特征天花板已明确。**精选和 Winsorize 是工程优化,不是 alpha 突破**。下一步应转向 **horizon 对齐**(中期反转策略),这才是真正能突破 RankIC 的方向。
