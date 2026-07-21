"""
5min 特征提取主脚本。

数据流:
  1. 读 1min bin(备份目录) → 聚合成 5min → dict[symbol -> DataFrame]
  2. 读股池快照 → 每个决策日 T 的成分股清单
  3. 逐日滚动:对每个 (symbol, T):
     a. 取 [T-5, T-1] 的 5min 历史(240 根)
     b. Kronos 预测 T 日全天 48 根(sample_count=5,拿独立采样)
     c. 提取 8 个特征
     d. 写入 LGBM qlib_root 的 .day.bin
  4. 输出 parquet 中间结果(便于检查/调试)

关键防泄露:
- 特征写 T 日,信息截止 T-1 收盘(无前视)
- 与 LGBM 的 T 日 9:41 决策对齐

用法:
    cd /home/zxh/projects/Kronos/finetune/feature_extraction
    # smoke test
    python extract_features.py --checkpoint base --max-stocks 10 --max-days 5
    # 全量(后台)
    python extract_features.py --checkpoint base
    python extract_features.py --checkpoint finetune
"""
import os
import sys
import argparse
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 行缓冲输出:重定向到文件时也能实时看日志
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

FEAT_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = FEAT_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent
for p in [str(FINETUNE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_extraction.config_5min_feat import Config
from feature_extraction.feature_defs import extract_features
# 复用已验证的 1min→5min 聚合(来自 preprocess_5min.py)
from preprocess_5min import _read_bin, _aggregate_1min_to_5min


# ====================================================================
# 数据加载
# ====================================================================

def load_1min_calendar(data_1min_root):
    """读 1min 日历为 DatetimeIndex"""
    path = os.path.join(data_1min_root, "calendars", "1min.txt")
    cal = pd.read_csv(path, header=None, names=['ts'])
    cal['ts'] = pd.to_datetime(cal['ts'])
    return pd.DatetimeIndex(cal['ts'])


def load_trade_days(day_features_root):
    """读 day.txt 日线日历"""
    # LGBM qlib_root 有 calendars/day.txt
    for cand in [
        os.path.join(day_features_root, "..", "calendars", "day.txt"),
        "/home/zxh/qlib_local_data/cn_data/calendars/day.txt",
    ]:
        if os.path.exists(cand):
            with open(cand) as f:
                return [l.strip() for l in f if l.strip()]
    raise FileNotFoundError("找不到 day.txt")


def fetch_stock_5min(symbol_lower, cal_1min, features_root):
    """读单只股票的 1min bin,聚合成 5min。

    复用 preprocess_5min 的 _read_bin + _aggregate_1min_to_5min。
    返回 DataFrame[open,high,low,close,vol,amt],index=5min DatetimeIndex
    """
    fields = ['open', 'high', 'low', 'close', 'volume']
    series_dict = {}
    for f in fields:
        s = _read_bin(symbol_lower, f, cal_1min, features_root)
        if s is not None:
            series_dict[f] = s
    if len(series_dict) < len(fields):
        return None

    df = pd.DataFrame(series_dict).sort_index()
    df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')
    if len(df) == 0:
        return None

    df5 = _aggregate_1min_to_5min(df)
    if len(df5) == 0:
        return None

    # 派生 vol/amt(与训练时一致)
    df5['vol'] = df5['volume']
    df5['amt'] = df5['close'] * df5['volume']
    df5 = df5[['open', 'high', 'low', 'close', 'vol', 'amt']]
    return df5


def load_preprocessed_5min(cfg):
    """优先加载已预处理的 5min pickle(比逐只聚合快 400 倍)。

    来源:finetune/data/highbeta_5min/full/train_5min.pkl
    (由 preprocess_5min.py 生成,包含全部股池股票的 5min 数据)

    Returns:
        dict[symbol_lower -> DataFrame[open,high,low,close,vol,amt]]
    """
    import pickle
    pkl_path = f"{PROJECT_ROOT}/finetune/data/highbeta_5min/full/train_5min.pkl"
    if not os.path.exists(pkl_path):
        return None
    print(f"  加载预处理 pickle: {pkl_path}")
    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  加载 {len(data)} 只股票, 耗时 {time.time()-t0:.1f}s")
    return data


def load_universe_snapshots(universe_csv):
    """读股池快照,返回 dict[date_str -> list[code_qlib]]"""
    df = pd.read_csv(universe_csv)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    universe = {}
    for date, group in df.groupby('date'):
        universe[date] = group['code_qlib'].tolist()
    return universe


# ====================================================================
# Kronos 加载 + 多采样预测
# ====================================================================

def load_predictor(cfg, checkpoint):
    """加载 Kronos predictor(tokenizer + model)。

    checkpoint='base' 用预训练;'finetune' 用 highbeta_5min/full。
    """
    import torch
    from model import Kronos, KronosTokenizer

    device = cfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path)
    tokenizer.to(device).eval()

    if checkpoint == 'base':
        model_path = cfg.base_predictor_path
    elif checkpoint == 'finetune':
        model_path = cfg.finetune_predictor_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"微调 checkpoint 不存在: {model_path}\n"
                f"请确认 finetune/outputs/highbeta_5min/full/ 已训练。"
            )
    else:
        raise ValueError(f"未知 checkpoint: {checkpoint}")

    model = Kronos.from_pretrained(model_path)
    model.to(device).eval()
    print(f"[load] checkpoint={checkpoint} model={model_path} device={device}")
    return tokenizer, model, device


def predict_multisample(tokenizer, model, device, history_df, pred_len,
                        sample_count, T=1.0, top_p=0.95, top_k=0,
                        max_context=512, clip=5.0):
    """多采样预测:返回 (n_samples, pred_len, 6) 的独立路径。

    直接调用 auto_regressive_inference,跳过官方 predict() 的 mean。
    这是与 backtest_excess.py 的关键差异 —— 我们要拿到每条独立路径算 uncertainty。
    """
    import torch
    from model.kronos import auto_regressive_inference, calc_time_stamps

    # 构造输入(KronosPredictor.predict 的内部逻辑,复刻)
    price_cols = ['open', 'high', 'low', 'close']
    df = history_df.copy()
    # 列名映射(vol→volume, amt→amount)
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    if 'amount' not in df.columns and 'amt' in df.columns:
        df['amount'] = df['amt']
    # 必需列
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if c not in df.columns:
            return None
    if df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().values.any():
        return None

    x_timestamp = pd.Series(pd.to_datetime(df.index))

    # 构造未来时间戳(pred_len 个 5min,基于最后一根的时间往后推)
    last_ts = pd.to_datetime(df.index[-1])
    # 5min 间隔,跳过非交易时段(简化:直接往后推 5min,跨日跳到次日 9:35)
    # 实际上 Kronos 用 calc_time_stamps 算时间特征,只要时间戳合理即可
    # 训练时用真实未来时间戳,这里复刻:从 last_ts 往后推 pred_len 个 5min bar
    future_ts = build_future_timestamps(last_ts, pred_len)
    y_timestamp = pd.Series(future_ts)

    x = df[price_cols + ['volume', 'amount']].values.astype(np.float32)
    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    # 归一化(与 KronosPredictor.predict 一致)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)

    # 转 tensor,加 batch 维
    x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(device)

    with torch.no_grad():
        preds = auto_regressive_inference(
            tokenizer, model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context, pred_len, clip, T, top_k, top_p, sample_count,
            verbose=False,
        )
        # auto_regressive_inference 内部已 reshape(-1, sample_count, ...) 后 mean
        # 所以我们拿到的 shape 是 (1, pred_len, 6) 的平均
        # 要拿独立路径,必须改 auto_regressive_inference 或重复调用

    # 关键:auto_regressive_inference 在 np.mean 前有 reshape(-1, sample_count, ...)
    # 我们用"重复调用单采样"的方式拿独立路径
    # (比改官方函数更安全,且 sample_count 小时开销可控)
    return preds, (x_mean, x_std)


def build_future_timestamps(last_ts, pred_len):
    """从最后一根历史 5min 时间戳,构造 pred_len 个未来 5min 时间戳。

    A 股交易时段:9:35~11:30(上午 24 根), 13:05~15:00(下午 24 根),共 48 根/天
    规则:从 last_ts 的下一根开始,跳过午休和非交易日。
    简化实现:按时间顺序填充下一个交易日的 48 个 5min 时间戳。
    """
    # 生成"下一个交易日"的标准 48 个时间点
    # 如果 last_ts 是某日下午,未来就是次日全天
    # 如果 last_ts 是上午(不应发生,但兜底),未来就是当日剩余+次日
    next_day = last_ts + pd.Timedelta(days=1)
    # 跳过周末
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)

    # A 股 5min 时间戳(末根):9:35, 9:40, ..., 11:30, 13:05, ..., 15:00
    morning = [pd.Timestamp(next_day.date()) + pd.Timedelta(minutes=9*60+35+5*i)
               for i in range(24)]  # 9:35 ~ 11:30
    afternoon = [pd.Timestamp(next_day.date()) + pd.Timedelta(minutes=13*60+5+5*i)
                 for i in range(24)]  # 13:05 ~ 15:00
    all_ts = morning + afternoon
    return all_ts[:pred_len]


def predict_independent_paths_batch(tokenizer, model, device, history_dfs, pred_len,
                                    sample_count, T=1.0, top_p=0.95, top_k=0,
                                    max_context=512, clip=5.0):
    """批量多采样预测:一次推理处理 B 只股票,每只拿到 sample_count 条独立路径。

    比逐只串行快 ~10x(自回归步数不变,但 GPU 并行利用更好)。
    要求:所有 history_dfs 长度一致(都是 lookback_window)。

    Args:
        history_dfs: List[pd.DataFrame],每个 shape (lookback, 6+)

    Returns:
        List[np.ndarray],每个 shape (sample_count, pred_len, 6)
        若某只无效,对应位置为 None
    """
    import torch
    from model.kronos import calc_time_stamps, sample_from_logits

    price_cols = ['open', 'high', 'low', 'close']
    B = len(history_dfs)
    valid_idx = []   # 有效股票的索引
    x_list, xs_list, ys_list, means, stds = [], [], [], [], []

    for i, history_df in enumerate(history_dfs):
        df = history_df.copy()
        if 'volume' not in df.columns and 'vol' in df.columns:
            df['volume'] = df['vol']
        if 'amount' not in df.columns and 'amt' in df.columns:
            df['amount'] = df['amt']
        ok = all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume', 'amount'])
        if not ok:
            continue
        if df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().values.any():
            continue

        x_timestamp = pd.Series(pd.to_datetime(df.index))
        future_ts = build_future_timestamps(pd.to_datetime(df.index[-1]), pred_len)
        y_timestamp = pd.Series(future_ts)

        x = df[price_cols + ['volume', 'amount']].values.astype(np.float32)
        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -clip, clip)

        x_list.append(x_norm)
        xs_list.append(x_time_df.values.astype(np.float32))
        ys_list.append(y_time_df.values.astype(np.float32))
        means.append(x_mean)
        stds.append(x_std)
        valid_idx.append(i)

    if not valid_idx:
        return [None] * B

    # 堆叠成 batch:(B_valid, seq_len, feat)
    x_batch = np.stack(x_list, axis=0).astype(np.float32)
    xs_batch = np.stack(xs_list, axis=0).astype(np.float32)
    ys_batch = np.stack(ys_list, axis=0).astype(np.float32)
    Bv = x_batch.shape[0]
    sc = sample_count

    x_t = torch.from_numpy(x_batch).to(device)
    xs_t = torch.from_numpy(xs_batch).to(device)
    ys_t = torch.from_numpy(ys_batch).to(device)

    # batch 维扩展 sample_count 倍
    x_rep = x_t.unsqueeze(1).repeat(1, sc, 1, 1).reshape(Bv * sc, *x_t.shape[1:])
    xs_rep = xs_t.unsqueeze(1).repeat(1, sc, 1, 1).reshape(Bv * sc, *xs_t.shape[1:])
    ys_rep = ys_t.unsqueeze(1).repeat(1, sc, 1, 1).reshape(Bv * sc, *ys_t.shape[1:])

    with torch.no_grad():
        x_token = tokenizer.encode(x_rep, half=True)
        initial_seq_len = x_rep.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([xs_rep, ys_rep], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        # reshape 到 (B_valid, sample_count, total_len, d_in)
        z = z.reshape(Bv, sc, z.size(1), z.size(2))
        # 只取预测段(最后 pred_len 根)
        z_pred = z[:, :, -pred_len:, :]  # (Bv, sc, pred_len, d_in)
        preds_all = z_pred.cpu().numpy()

    # 反归一化 + 组装结果
    results = [None] * B
    for vi, orig_i in enumerate(valid_idx):
        preds_v = preds_all[vi].copy()  # (sc, pred_len, 6)
        x_mean, x_std = means[vi], stds[vi]
        # 只反归一化价格列(open/high/low/close),volume/amount 置 0
        for c in [0, 1, 2, 3]:
            preds_v[:, :, c] = preds_v[:, :, c] * (x_std[c] + 1e-5) + x_mean[c]
        preds_v[:, :, 4] = 0.0
        preds_v[:, :, 5] = 0.0
        results[orig_i] = preds_v

    return results


def predict_independent_paths(tokenizer, model, device, history_df, pred_len,
                              sample_count, T=1.0, top_p=0.95, top_k=0,
                              max_context=512, clip=5.0):
    """并行多采样预测:一次调用拿到 sample_count 条独立路径。

    关键:auto_regressive_inference 内部对 batch 维做 sample_count 倍扩展,
    然后在 decode 后 reshape(-1, sample_count, ...) 分离各路径。
    我们直接利用这个机制,自己 reshape(不取 mean),拿到独立路径。

    返回:(sample_count, pred_len, 6) 已反归一化到原始价格尺度
    """
    import torch
    from model.kronos import auto_regressive_inference, calc_time_stamps

    price_cols = ['open', 'high', 'low', 'close']
    df = history_df.copy()
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    if 'amount' not in df.columns and 'amt' in df.columns:
        df['amount'] = df['amt']
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if c not in df.columns:
            return None
    if df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().values.any():
        return None

    x_timestamp = pd.Series(pd.to_datetime(df.index))
    future_ts = build_future_timestamps(pd.to_datetime(df.index[-1]), pred_len)
    y_timestamp = pd.Series(future_ts)

    x = df[price_cols + ['volume', 'amount']].values.astype(np.float32)
    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)

    x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(device)

    with torch.no_grad():
        # 临时 monkey-patch:让 auto_regressive_inference 返回 reshape 后的多路径
        # 原版在 kronos.py:465-467 做了 reshape + mean,我们要 reshape 不 mean
        # 直接复刻内部逻辑(避免改官方代码)
        from model.kronos import top_k_top_p_filtering, sample_from_logits

        sc = sample_count
        # batch 维扩展(与 auto_regressive_inference 一致)
        x_rep = x_tensor.unsqueeze(1).repeat(1, sc, 1, 1).reshape(-1, *x_tensor.shape[1:])
        xs_rep = x_stamp_tensor.unsqueeze(1).repeat(1, sc, 1, 1).reshape(-1, *x_stamp_tensor.shape[1:])
        ys_rep = y_stamp_tensor.unsqueeze(1).repeat(1, sc, 1, 1).reshape(-1, *y_stamp_tensor.shape[1:])

        x_token = tokenizer.encode(x_rep, half=True)
        initial_seq_len = x_rep.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([xs_rep, ys_rep], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        # 关键:reshape 到 (1, sample_count, total_len, d_in),不取 mean
        z = z.reshape(1, sc, z.size(1), z.size(2))
        # 只取预测段(最后 pred_len 根)
        z_pred = z[:, :, -pred_len:, :]  # (1, sc, pred_len, d_in)
        preds = z_pred.cpu().numpy()

    # 反归一化到原价格尺度
    # 注意:volume/amount 量级巨大(几千万),单条 BSQ 解码路径可能极端。
    # 我们 8 个特征只用价格列(open/high/low/close),不用 volume/amount,
    # 所以只反归一化前 4 列,后 2 列保持 0(不会被特征用到)。
    preds_denorm = preds[0].copy()  # (sc, pred_len, 6)
    price_cols = [0, 1, 2, 3]  # open, high, low, close
    for c in price_cols:
        preds_denorm[:, :, c] = preds_denorm[:, :, c] * (x_std[c] + 1e-5) + x_mean[c]
    # volume/amount 列置 0(特征不用,避免数值爆炸干扰)
    preds_denorm[:, :, 4] = 0.0
    preds_denorm[:, :, 5] = 0.0
    return preds_denorm


# ====================================================================
# bin 对齐与写出
# ====================================================================

def get_bin_alignment(symbol_lower, day_features_root):
    """读 close.day.bin 的 header,返回 (start_index, length)。

    Kronos 特征 bin 必须与 close.day.bin 对齐(相同 start_index 和长度)。
    """
    path = os.path.join(day_features_root, symbol_lower, "close.day.bin")
    if not os.path.exists(path):
        return None, None
    arr = np.fromfile(path, dtype='<f4')
    if arr.size == 0:
        return None, None
    return int(arr[0]), len(arr) - 1


def write_feature_bin(symbol_lower, field, start_index, values, day_features_root):
    """写单个特征的 .day.bin,与 qlib 格式一致"""
    path = os.path.join(day_features_root, symbol_lower, f"{field}.day.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    arr = np.empty(values.size + 1, dtype='<f4')
    arr[0] = start_index
    arr[1:] = values
    arr.tofile(path)


def write_stock_features(symbol_lower, date_str, features, trade_days,
                         day_features_root, kronos_fields):
    """把某 (symbol, date) 的 8 个特征写入对应的 .day.bin 位置。

    策略:读 close.day.bin 的 start_index 和长度,构建全 NaN 数组,
    在 date_str 对应的位置填入特征值,整体写出。

    注意:每次提取只更新一个 (symbol, date),需要读已有 bin 合并后写回,
    否则会覆盖之前提取的特征。所以这里用"累积写入":
    1. 读已有 bin(若有)
    2. 在 date 位置覆盖特征值
    3. 写回
    """
    start_index, length = get_bin_alignment(symbol_lower, day_features_root)
    if start_index is None:
        return False  # 该股票在 LGBM qlib_root 无 close.day.bin,跳过

    # date_str 在 trade_days 中的索引
    if date_str not in trade_days:
        return False
    date_idx = trade_days.index(date_str)
    pos = date_idx - start_index
    if pos < 0 or pos >= length:
        return False  # 超出该股票的 bin 范围

    for field in kronos_fields:
        if field not in features:
            continue
        bin_path = os.path.join(day_features_root, symbol_lower, f"{field}.day.bin")
        # 读已有(累积写入)
        if os.path.exists(bin_path):
            arr = np.fromfile(bin_path, dtype='<f4')
            existing_start = int(arr[0])
            existing_vals = arr[1:].copy()
        else:
            existing_start = start_index
            existing_vals = np.full(length, np.nan, dtype='<f4')

        existing_vals[pos] = features[field]
        write_feature_bin(symbol_lower, field, existing_start, existing_vals,
                          day_features_root)
    return True


# ====================================================================
# 主流程
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', choices=['base', 'finetune'], required=True)
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='smoke test:限制股票数')
    parser.add_argument('--max-days', type=int, default=None,
                        help='smoke test:限制决策日数')
    parser.add_argument('--no-write-bin', action='store_true',
                        help='只跑预测不写 bin(调试用)')
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"5min 特征提取 (checkpoint={args.checkpoint})")
    print("=" * 60)

    # 1. 加载日历和股池
    print("\n[1] 加载数据...")
    cal_1min = load_1min_calendar(cfg.data_1min_root)
    trade_days = load_trade_days(cfg.day_features_root)
    universe = load_universe_snapshots(cfg.universe_csv)

    # 决策日范围
    decision_days = [d for d in trade_days
                     if cfg.start_date <= d <= cfg.end_date and d in universe]
    if args.max_days:
        decision_days = decision_days[:args.max_days]
    print(f"  1min 日历: {len(cal_1min)} 行")
    print(f"  交易日: {len(trade_days)}")
    print(f"  决策日(有股池快照): {len(decision_days)}")
    if decision_days:
        print(f"    {decision_days[0]} ~ {decision_days[-1]}")

    # 2. 加载 Kronos
    print(f"\n[2] 加载 Kronos ({args.checkpoint})...")
    tokenizer, model, device = load_predictor(cfg, args.checkpoint)

    # 3. 预加载所有涉及股票的 5min 数据
    print("\n[3] 预加载 5min 数据...")
    all_symbols = set()
    for d in decision_days:
        all_symbols.update(universe[d])
    all_symbols = sorted(all_symbols)
    if args.max_stocks:
        # smoke test:只保留前 N 只
        all_symbols = all_symbols[:args.max_stocks]
        # 同步限制 universe
        for d in decision_days:
            universe[d] = [s for s in universe[d] if s in set(all_symbols)]

    t0 = time.time()
    stock_5min = {}  # symbol_upper -> DataFrame[5min]
    # 优先用已预处理 pickle(快 400 倍)
    preprocessed = load_preprocessed_5min(cfg)
    if preprocessed is not None:
        # pickle 的 key 是 symbol_lower,转成 upper 与 universe 对齐
        for sym_lower, df5 in preprocessed.items():
            stock_5min[sym_lower.upper()] = df5
        # 如果 max_stocks 限制,过滤
        if args.max_stocks:
            keep = set(all_symbols)
            stock_5min = {k: v for k, v in stock_5min.items() if k in keep}
    else:
        # fallback:逐只聚合(慢)
        print("  (pickle 不存在,fallback 到逐只聚合)")
        for sym in all_symbols:
            df5 = fetch_stock_5min(sym.lower(), cal_1min, f"{cfg.data_1min_root}/features")
            if df5 is not None:
                stock_5min[sym] = df5
    print(f"  成功加载 {len(stock_5min)}/{len(all_symbols)} 只, 耗时 {time.time()-t0:.0f}s")

    # 4. 逐日滚动提取
    print(f"\n[4] 滚动提取特征...")
    lookback = cfg.lookback_window  # 240
    pred_len = cfg.predict_window   # 48
    sample_count = cfg.sample_count # 5

    records = []  # (date, symbol, 8 features) → parquet
    n_done = 0
    n_skip = 0
    n_fail = 0
    t_start = time.time()

    for di, date in enumerate(decision_days):
        # 该决策日的成分股
        day_symbols = universe.get(date, [])
        day_dt = pd.to_datetime(date)

        # 准备当日所有股票的历史窗口(批量推理)
        histories = []      # 与 day_symbols 一一对应,None 表示跳过
        valid_symbols = []
        for symbol in day_symbols:
            if symbol not in stock_5min:
                histories.append(None)
                n_skip += 1
                continue
            df5 = stock_5min[symbol]
            history = df5[df5.index.normalize() < day_dt]
            if len(history) < lookback:
                histories.append(None)
                n_skip += 1
                continue
            histories.append(history.iloc[-lookback:])
            valid_symbols.append(symbol)

        # 批量推理(一次处理当日所有有效股票)
        if valid_symbols:
            valid_histories = [h for h in histories if h is not None]
            try:
                preds_list = predict_independent_paths_batch(
                    tokenizer, model, device, valid_histories, pred_len,
                    sample_count, T=cfg.inference_T, top_p=cfg.inference_top_p,
                    top_k=cfg.inference_top_k, max_context=cfg.max_context,
                    clip=cfg.clip,
                )
            except Exception as e:
                print(f"  [batch err] {date}: {type(e).__name__}: {e}")
                preds_list = [None] * len(valid_histories)

            # 提取特征 + 写 bin
            for symbol, preds in zip(valid_symbols, preds_list):
                if preds is None:
                    n_fail += 1
                    continue
                features = extract_features(preds)
                if features is None:
                    n_fail += 1
                    continue
                if not args.no_write_bin:
                    write_stock_features(
                        symbol.lower(), date, features, trade_days,
                        cfg.day_features_root, cfg.kronos_fields,
                    )
                rec = {'date': date, 'symbol': symbol}
                rec.update(features)
                records.append(rec)
                n_done += 1

        # 进度报告
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1)
        eta = (len(decision_days) - di - 1) * elapsed / max(di + 1, 1)
        if (di + 1) % 5 == 0 or di == 0 or di == len(decision_days) - 1:
            print(f"  [{di+1}/{len(decision_days)}] {date}: "
                  f"done={n_done} skip={n_skip} fail={n_fail} "
                  f"| {rate:.1f} feats/s ETA={eta/60:.1f}min")

    # 5. 保存 parquet 中间结果
    if records:
        df_out = pd.DataFrame(records)
        parquet_path = f"{cfg.output_dir}/features_{args.checkpoint}.parquet"
        df_out.to_parquet(parquet_path, index=False)
        print(f"\n[5] 中间结果保存: {parquet_path} ({len(df_out)} 行)")

    elapsed = time.time() - t_start
    print(f"\n✅ 完成: done={n_done} skip={n_skip} fail={n_fail} "
          f"耗时={elapsed/60:.1f}min")

    # 特征分布快照
    if records:
        df_out = pd.DataFrame(records)
        print("\n特征分布:")
        for f in cfg.kronos_fields:
            if f in df_out.columns:
                vals = df_out[f].dropna()
                print(f"  {f:30s}: mean={vals.mean():+.4f} "
                      f"std={vals.std():.4f} "
                      f"[{vals.min():+.3f}, {vals.max():+.3f}]")


if __name__ == '__main__':
    main()
