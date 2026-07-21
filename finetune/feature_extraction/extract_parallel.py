"""
多卡并行特征提取。

把 598 个决策日分给 N 张 GPU,每张 GPU 跑一个 worker 进程,各处理一部分决策日。
各 worker 写到独立临时目录(避免 bin 竞争),最后合并到 LGBM qlib_root。

用法:
    cd /home/zxh/projects/Kronos/finetune/feature_extraction
    python extract_parallel.py --checkpoint base --gpus 1,2,3,4,5,6
    python extract_parallel.py --checkpoint finetune --gpus 1,2,3,4,5,6

设计:
- 每个 worker 绑定一张 GPU(CUDA_VISIBLE_DEVICES)
- 决策日按 shard_id 分片(worker 0 处理 day[0::N], worker 1 处理 day[1::N]...)
- 每个 worker 加载完整 5min pickle(3 秒,可接受)
- 每个 worker 写到独立 parquet(output/shard_{id}.parquet)
- bin 写入由 worker 直接写到 qlib_root(不同决策日写不同位置,无冲突)
  ⚠️ 注意:同一只股票的 bin 会被多 worker 写,但写的是不同 date 位置,
  累积写入逻辑(read-modify-write)有竞争风险,需加文件锁。

为避免 bin 竞争,本实现用两阶段:
  阶段 1:各 worker 只写 parquet(不写 bin),快速并行
  阶段 2:单进程读所有 parquet,统一写 bin(无竞争)

这样既快又安全。
"""
import os
import sys
import argparse
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

FEAT_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = FEAT_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent
for p in [str(FINETUNE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from feature_extraction.config_5min_feat import Config
from feature_extraction.feature_defs import extract_features
from preprocess_5min import _read_bin, _aggregate_1min_to_5min


def load_1min_calendar(data_1min_root):
    path = os.path.join(data_1min_root, "calendars", "1min.txt")
    cal = pd.read_csv(path, header=None, names=['ts'])
    cal['ts'] = pd.to_datetime(cal['ts'])
    return pd.DatetimeIndex(cal['ts'])


def load_trade_days(day_features_root):
    for cand in [
        os.path.join(day_features_root, "..", "calendars", "day.txt"),
        "/home/zxh/qlib_local_data/cn_data/calendars/day.txt",
    ]:
        if os.path.exists(cand):
            with open(cand) as f:
                return [l.strip() for l in f if l.strip()]
    raise FileNotFoundError("找不到 day.txt")


def load_universe_snapshots(universe_csv):
    df = pd.read_csv(universe_csv)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    universe = {}
    for date, group in df.groupby('date'):
        universe[date] = group['code_qlib'].tolist()
    return universe


def fetch_stock_5min(symbol_lower, cal_1min, features_root):
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
    df5['vol'] = df5['volume']
    df5['amt'] = df5['close'] * df5['volume']
    df5 = df5[['open', 'high', 'low', 'close', 'vol', 'amt']]
    return df5


def load_preprocessed_5min(cfg):
    pkl_path = f"{PROJECT_ROOT}/finetune/data/highbeta_5min/full/train_5min.pkl"
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_predictor(cfg, checkpoint, device):
    import torch
    from model import Kronos, KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path)
    tokenizer.to(device).eval()
    if checkpoint == 'base':
        model_path = cfg.base_predictor_path
    else:
        model_path = cfg.finetune_predictor_path
    model = Kronos.from_pretrained(model_path)
    model.to(device).eval()
    return tokenizer, model


def build_future_timestamps(last_ts, pred_len):
    import pandas as pd
    next_day = last_ts + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    morning = [pd.Timestamp(next_day.date()) + pd.Timedelta(minutes=9*60+35+5*i)
               for i in range(24)]
    afternoon = [pd.Timestamp(next_day.date()) + pd.Timedelta(minutes=13*60+5+5*i)
                 for i in range(24)]
    all_ts = morning + afternoon
    return all_ts[:pred_len]


def predict_batch_multisample(tokenizer, model, device, history_dfs, pred_len,
                              sample_count, T=1.0, top_p=0.95, top_k=0,
                              max_context=512, clip=5.0):
    """批量多采样预测,返回 List[np.ndarray(sample_count, pred_len, 6)]"""
    import torch
    from model.kronos import calc_time_stamps, sample_from_logits

    price_cols = ['open', 'high', 'low', 'close']
    B = len(history_dfs)
    valid_idx, x_list, xs_list, ys_list, means, stds = [], [], [], [], [], []

    for i, history_df in enumerate(history_dfs):
        df = history_df.copy()
        if 'volume' not in df.columns and 'vol' in df.columns:
            df['volume'] = df['vol']
        if 'amount' not in df.columns and 'amt' in df.columns:
            df['amount'] = df['amt']
        ok = all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume', 'amount'])
        if not ok or df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().values.any():
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

    x_batch = np.stack(x_list, axis=0).astype(np.float32)
    xs_batch = np.stack(xs_list, axis=0).astype(np.float32)
    ys_batch = np.stack(ys_list, axis=0).astype(np.float32)
    Bv = x_batch.shape[0]
    sc = sample_count

    x_t = torch.from_numpy(x_batch).to(device)
    xs_t = torch.from_numpy(xs_batch).to(device)
    ys_t = torch.from_numpy(ys_batch).to(device)
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
        z = z.reshape(Bv, sc, z.size(1), z.size(2))
        z_pred = z[:, :, -pred_len:, :]
        preds_all = z_pred.cpu().numpy()

    results = [None] * B
    for vi, orig_i in enumerate(valid_idx):
        preds_v = preds_all[vi].copy()
        x_mean, x_std = means[vi], stds[vi]
        for c in [0, 1, 2, 3]:
            preds_v[:, :, c] = preds_v[:, :, c] * (x_std[c] + 1e-5) + x_mean[c]
        preds_v[:, :, 4] = 0.0
        preds_v[:, :, 5] = 0.0
        results[orig_i] = preds_v
    return results


def worker(shard_id, num_shards, gpu_id, checkpoint,
           decision_days, universe, stock_5min, cfg_dict):
    """单个 worker 进程:处理 decision_days[shard_id::num_shards]"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    device = 'cuda'

    cfg = Config()
    my_days = decision_days[shard_id::num_shards]
    print(f"[shard {shard_id}/{num_shards} GPU {gpu_id}] 处理 {len(my_days)} 天: "
          f"{my_days[0]} ~ {my_days[-1]}", flush=True)

    tokenizer, model = load_predictor(cfg, checkpoint, device)

    records = []
    n_done, n_skip, n_fail = 0, 0, 0
    t_start = time.time()
    lookback = cfg.lookback_window
    pred_len = cfg.predict_window
    sc = cfg.sample_count

    for di, date in enumerate(my_days):
        day_symbols = universe.get(date, [])
        day_dt = pd.to_datetime(date)

        histories = []
        valid_symbols = []
        for symbol in day_symbols:
            if symbol not in stock_5min:
                n_skip += 1
                continue
            df5 = stock_5min[symbol]
            history = df5[df5.index.normalize() < day_dt]
            if len(history) < lookback:
                n_skip += 1
                continue
            histories.append(history.iloc[-lookback:])
            valid_symbols.append(symbol)

        if valid_symbols:
            try:
                preds_list = predict_batch_multisample(
                    tokenizer, model, device, histories, pred_len,
                    sc, T=cfg.inference_T, top_p=cfg.inference_top_p,
                    top_k=cfg.inference_top_k, max_context=cfg.max_context, clip=cfg.clip,
                )
            except Exception as e:
                print(f"[shard {shard_id}] batch err {date}: {e}", flush=True)
                preds_list = [None] * len(valid_symbols)

            for symbol, preds in zip(valid_symbols, preds_list):
                if preds is None:
                    n_fail += 1
                    continue
                features = extract_features(preds)
                if features is None:
                    n_fail += 1
                    continue
                rec = {'date': date, 'symbol': symbol}
                rec.update(features)
                records.append(rec)
                n_done += 1

        if (di + 1) % 10 == 0 or di == 0 or di == len(my_days) - 1:
            elapsed = time.time() - t_start
            eta = (len(my_days) - di - 1) * elapsed / max(di + 1, 1)
            print(f"[shard {shard_id} GPU {gpu_id}] [{di+1}/{len(my_days)}] {date}: "
                  f"done={n_done} skip={n_skip} fail={n_fail} ETA={eta/60:.1f}min", flush=True)

    # 保存该 shard 的 parquet
    out_path = f"{cfg.output_dir}/shard_{shard_id}_{checkpoint}.parquet"
    if records:
        pd.DataFrame(records).to_parquet(out_path, index=False)
    elapsed = time.time() - t_start
    print(f"[shard {shard_id} GPU {gpu_id}] ✅ 完成 done={n_done} 耗时={elapsed/60:.1f}min "
          f"→ {out_path}", flush=True)
    return shard_id, n_done, out_path


def write_bins_from_parquet(parquet_paths, cfg, trade_days):
    """阶段 2:读所有 shard parquet,统一写 bin(无竞争)"""
    print("\n=== 阶段 2:统一写 bin ===")
    all_records = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        all_records.append(df)
        print(f"  {p}: {len(df)} 行")
    df_all = pd.concat(all_records, ignore_index=True)
    print(f"  合计: {len(df_all)} 行")

    # 按股票分组,逐只股票写完整 bin(避免跨 shard 的 read-modify-write)
    kronos_fields = cfg.kronos_fields
    n_written = 0
    n_skip = 0
    for symbol, group in df_all.groupby('symbol'):
        sym_lower = symbol.lower()
        # 读 close.day.bin 的对齐信息
        path = os.path.join(cfg.day_features_root, sym_lower, "close.day.bin")
        if not os.path.exists(path):
            n_skip += len(group)
            continue
        arr = np.fromfile(path, dtype='<f4')
        if arr.size == 0:
            n_skip += len(group)
            continue
        start_index = int(arr[0])
        length = len(arr) - 1

        # 为每个特征构建全 NaN 数组,填入该股票的所有记录
        for field in kronos_fields:
            vals = np.full(length, np.nan, dtype=np.float32)
            for _, row in group.iterrows():
                if row['date'] in trade_days:
                    date_idx = trade_days.index(row['date'])
                    pos = date_idx - start_index
                    if 0 <= pos < length and field in row:
                        vals[pos] = row[field]
            # 写 bin
            out_path = os.path.join(cfg.day_features_root, sym_lower, f"{field}.day.bin")
            out_arr = np.empty(vals.size + 1, dtype='<f4')
            out_arr[0] = start_index
            out_arr[1:] = vals
            out_arr.tofile(out_path)
        n_written += len(group)

    print(f"  ✅ 写入 {n_written} 条记录,跳过 {n_skip} 条(无 close.day.bin)")
    return df_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', choices=['base', 'finetune'], required=True)
    parser.add_argument('--gpus', type=str, default='1,2,3,4,5,6',
                        help='使用的 GPU id 列表,逗号分隔')
    parser.add_argument('--merge-only', action='store_true',
                        help='只做阶段 2(合并已有 shard parquet 写 bin)')
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(',')]
    num_shards = len(gpus)

    if args.merge_only:
        # 只合并
        trade_days = load_trade_days(cfg.day_features_root)
        parquet_paths = sorted([
            f"{cfg.output_dir}/shard_{i}_{args.checkpoint}.parquet"
            for i in range(num_shards)
            if os.path.exists(f"{cfg.output_dir}/shard_{i}_{args.checkpoint}.parquet")
        ])
        df_all = write_bins_from_parquet(parquet_paths, cfg, trade_days)
        # 保存合并的完整 parquet
        df_all.to_parquet(f"{cfg.output_dir}/features_{args.checkpoint}.parquet", index=False)
        # 特征分布
        print("\n特征分布:")
        for f in cfg.kronos_fields:
            if f in df_all.columns:
                vals = df_all[f].dropna()
                print(f"  {f:30s}: mean={vals.mean():+.4f} std={vals.std():.4f}")
        return

    print("=" * 60)
    print(f"多卡并行特征提取 (checkpoint={args.checkpoint}, GPUs={gpus})")
    print("=" * 60)

    # 1. 加载数据(主进程)
    print("\n[1] 加载数据...")
    cal_1min = load_1min_calendar(cfg.data_1min_root)
    trade_days = load_trade_days(cfg.day_features_root)
    universe = load_universe_snapshots(cfg.universe_csv)
    decision_days = [d for d in trade_days
                     if cfg.start_date <= d <= cfg.end_date and d in universe]
    print(f"  决策日: {len(decision_days)} ({decision_days[0]} ~ {decision_days[-1]})")

    # 2. 加载 5min 数据(共享给所有 worker)
    print("\n[2] 加载 5min 数据...")
    t0 = time.time()
    preprocessed = load_preprocessed_5min(cfg)
    stock_5min = {}
    if preprocessed:
        for sym_lower, df5 in preprocessed.items():
            stock_5min[sym_lower.upper()] = df5
    print(f"  加载 {len(stock_5min)} 只, 耗时 {time.time()-t0:.1f}s")

    # 3. 清理旧 shard parquet
    for i in range(num_shards):
        p = f"{cfg.output_dir}/shard_{i}_{args.checkpoint}.parquet"
        if os.path.exists(p):
            os.remove(p)
            print(f"  清理旧 shard: {p}")

    # 4. 启动 worker 进程池
    print(f"\n[3] 启动 {num_shards} 个 worker (GPU {gpus})...")
    t_start = time.time()
    completed = []

    # 用 subprocess 而非 ProcessPoolExecutor,避免 CUDA fork 问题
    processes = []
    log_files = []
    for shard_id, gpu_id in enumerate(gpus):
        log_path = f"{cfg.output_dir}/logs/shard_{shard_id}_gpu{gpu_id}_{args.checkpoint}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, 'w')
        log_files.append(log_f)

        cmd = [
            sys.executable, __file__,
            '--worker', str(shard_id), str(num_shards), str(gpu_id),
            '--checkpoint', args.checkpoint,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['HF_HUB_OFFLINE'] = '1'
        env['PYTHONPATH'] = f"{FINETUNE_DIR}:{PROJECT_ROOT}"
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append(p)
        print(f"  shard {shard_id} → GPU {gpu_id} (PID {p.pid}, log: {log_path})")

    # 等待所有 worker
    print(f"\n[4] 等待 {num_shards} 个 worker 完成...")
    for p in processes:
        p.wait()
        print(f"  PID {p.pid} 退出码 {p.returncode}")
    for f in log_files:
        f.close()

    elapsed = time.time() - t_start
    print(f"\n所有 worker 完成,总耗时 {elapsed/60:.1f}min")

    # 5. 合并写 bin
    trade_days = load_trade_days(cfg.day_features_root)
    parquet_paths = sorted([
        f"{cfg.output_dir}/shard_{i}_{args.checkpoint}.parquet"
        for i in range(num_shards)
        if os.path.exists(f"{cfg.output_dir}/shard_{i}_{args.checkpoint}.parquet")
    ])
    df_all = write_bins_from_parquet(parquet_paths, cfg, trade_days)
    df_all.to_parquet(f"{cfg.output_dir}/features_{args.checkpoint}.parquet", index=False)

    print("\n特征分布:")
    for f in cfg.kronos_fields:
        if f in df_all.columns:
            vals = df_all[f].dropna()
            print(f"  {f:30s}: mean={vals.mean():+.4f} std={vals.std():.4f} "
                  f"[{vals.min():+.3f}, {vals.max():+.3f}]")

    print(f"\n✅ 全部完成,总耗时 {elapsed/60:.1f}min")


def worker_main():
    """worker 进程入口"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=3, metavar=('SHARD_ID', 'NUM_SHARDS', 'GPU_ID'))
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    shard_id = int(args.worker[0])
    num_shards = int(args.worker[1])
    gpu_id = int(args.worker[2])

    cfg = Config()
    cal_1min = load_1min_calendar(cfg.data_1min_root)
    trade_days = load_trade_days(cfg.day_features_root)
    universe = load_universe_snapshots(cfg.universe_csv)
    decision_days = [d for d in trade_days
                     if cfg.start_date <= d <= cfg.end_date and d in universe]

    t0 = time.time()
    preprocessed = load_preprocessed_5min(cfg)
    stock_5min = {}
    if preprocessed:
        for sym_lower, df5 in preprocessed.items():
            stock_5min[sym_lower.upper()] = df5
    print(f"[shard {shard_id}] 加载 {len(stock_5min)} 只, 耗时 {time.time()-t0:.1f}s", flush=True)

    worker(shard_id, num_shards, gpu_id, args.checkpoint,
           decision_days, universe, stock_5min, None)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker_main()
    else:
        main()
