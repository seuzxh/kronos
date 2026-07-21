"""
8 个 Kronos 5min 特征定义。

输入:多次采样的预测 K 线路径,shape (n_samples, 48, 6)
      列顺序: [open, high, low, close, volume, amount]
输出:8 个标量特征

设计原则:
- 与 LGBM 的 1.5 天 label horizon 对齐(预测 T 日全天)
- 不只用"收益",而是从预测路径中提取多维信息
- kronos_uncertainty 来自采样间方差,反映模型对该股的信心
"""
import numpy as np


def extract_features(preds_multisample):
    """从多次采样的预测路径提取 8 个特征。

    Args:
        preds_multisample: np.ndarray, shape (n_samples, 48, 6)
            n_samples = sample_count(默认 5)
            48 = predict_window(次日全天 48 根 5min)
            6 = [open, high, low, close, volume, amount]

    Returns:
        dict: 8 个标量特征(key 与 config.kronos_fields 对应)
              若输入无效返回 None
    """
    if preds_multisample is None or len(preds_multisample) == 0:
        return None

    # 统一 shape
    if preds_multisample.ndim == 2:
        # 单次采样 (48, 6) → 扩维
        preds_multisample = preds_multisample[np.newaxis, ...]
    n_samples, n_steps, n_cols = preds_multisample.shape
    if n_steps < 2 or n_cols < 4:
        return None

    # 列索引(与 Kronos feature_list 一致)
    OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3

    features = {}

    # 对每次采样先算 per-sample 标量,再跨采样聚合
    # 这样 uncertainty 才有意义(基于同一指标的采样间方差)
    per_sample_rets = []        # 用于 uncertainty
    per_sample_intraday = {}    # 其他特征的 per-sample 值

    for s in range(n_samples):
        path = preds_multisample[s]  # (48, 6)

        # 当日开盘 = 第 0 根的 open;收盘 = 末根的 close
        day_open = float(path[0, OPEN])
        day_close = float(path[-1, CLOSE])
        if day_open <= 0 or np.isnan(day_open) or np.isnan(day_close):
            return None

        # 1. 日内累计收益(收/开 - 1)
        ret_intraday = day_close / day_open - 1
        per_sample_rets.append(ret_intraday)

        # 2. 日内最高涨幅(max_high / open - 1)
        max_high = float(np.max(path[:, HIGH]))
        high_ratio = max_high / day_open - 1

        # 3. 日内最大跌幅(min_low / open - 1,负数)
        min_low = float(np.min(path[:, LOW]))
        low_ratio = min_low / day_open - 1

        # 4. 振幅 (max_high - min_low) / open
        range_val = (max_high - min_low) / day_open

        # 5. 收盘相对位置 (收-低)/(高-低),∈ [0, 1]
        #    >0.5 = 收在高位(强势),<0.5 = 收在低位(弱势)
        #    注意:BSQ 解码异常时 close 可能超出 [low, high],需 clip 到 [0, 1]
        denom = max_high - min_low
        if denom > 1e-9:
            close_pos = (day_close - min_low) / denom
            close_pos = float(np.clip(close_pos, 0.0, 1.0))  # 防越界
        else:
            close_pos = 0.5  # 无波动,中性

        # 6/7. 上午/下午收益
        # A 股 5min:48 根 = 上午 24 根(9:35-11:30) + 下午 24 根(13:05-15:00)
        # 上午收盘 ≈ 第 23 根(11:30),下午开盘 ≈ 第 24 根(13:05)
        mid_idx = min(24, n_steps - 1)
        morning_close = float(path[mid_idx - 1, CLOSE])  # 上午末根收盘
        afternoon_close = float(path[-1, CLOSE])         # 下午末根收盘
        # 上午收益:从开盘到上午收盘
        morning_ret = morning_close / day_open - 1 if day_open > 0 else 0
        # 下午收益:从上午收盘到下午收盘
        afternoon_ret = (afternoon_close / morning_close - 1
                         if morning_close > 0 else 0)

        per_sample_intraday.setdefault('high_ratio', []).append(high_ratio)
        per_sample_intraday.setdefault('low_ratio', []).append(low_ratio)
        per_sample_intraday.setdefault('range', []).append(range_val)
        per_sample_intraday.setdefault('close_pos', []).append(close_pos)
        per_sample_intraday.setdefault('morning_ret', []).append(morning_ret)
        per_sample_intraday.setdefault('afternoon_ret', []).append(afternoon_ret)

    # 跨采样取平均(主特征)
    per_sample_rets = np.array(per_sample_rets)
    features['kronos_ret_intraday'] = float(np.mean(per_sample_rets))
    for k, vals in per_sample_intraday.items():
        features[f'kronos_{k}'] = float(np.mean(vals))

    # 8. uncertainty:多次采样间收益的标准差
    #    用样本标准差(ddof=1)更稳;样本数=1 时记 0
    if n_samples >= 2:
        features['kronos_uncertainty'] = float(np.std(per_sample_rets, ddof=1))
    else:
        features['kronos_uncertainty'] = 0.0

    return features


def self_test():
    """自检:人造路径验证特征取值合理"""
    np.random.seed(0)

    # Case 1: 单边上涨路径(每根 close 比 open 高 0.5%)
    OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3
    path_up = np.zeros((48, 6))
    path_up[:, OPEN] = 10.0
    for i in range(48):
        path_up[i, HIGH] = 10.0 * (1 + 0.005 * (i + 1) + 0.001)
        path_up[i, LOW] = 10.0 * (1 + 0.005 * (i + 1) - 0.001)
        path_up[i, CLOSE] = 10.0 * (1 + 0.005 * (i + 1))
    # 5 次相同采样
    preds_up = np.stack([path_up] * 5)
    f_up = extract_features(preds_up)
    print("Case 1 单边上涨:")
    for k, v in f_up.items():
        print(f"  {k}: {v:+.4f}")
    assert f_up['kronos_ret_intraday'] > 0.2, "单边上涨收益应 >20%"
    assert f_up['kronos_close_pos'] > 0.9, "单边上涨收盘应在高位"
    assert f_up['kronos_uncertainty'] < 1e-6, "相同采样 uncertainty 应≈0"

    # Case 2: 高 uncertainty(5 条路径收益差异大)
    paths = []
    for r in [0.05, -0.03, 0.02, -0.01, 0.04]:
        p = path_up.copy()
        p[:, CLOSE] = p[:, CLOSE] * (1 + r) / p[:, CLOSE][-1] * p[:, CLOSE][-1]
        # 简化:直接缩放 close
        scale = (1 + r)
        p[:, CLOSE] = path_up[:, CLOSE] * scale
        p[:, HIGH] = path_up[:, HIGH] * scale
        p[:, LOW] = path_up[:, LOW] * scale
        paths.append(p)
    preds_var = np.stack(paths)
    f_var = extract_features(preds_var)
    print("\nCase 2 高分歧:")
    for k, v in f_var.items():
        print(f"  {k}: {v:+.4f}")
    assert f_var['kronos_uncertainty'] > 0.02, "分歧大 uncertainty 应 >2%"

    print("\n✅ 特征定义自检通过")


if __name__ == '__main__':
    self_test()
