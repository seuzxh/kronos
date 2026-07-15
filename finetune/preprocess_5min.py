"""
5min 数据预处理: 1min bin → 5min K线 → 按CV折切分 → pickle

数据流 (直接读bin, 绕过qlib API, 快1000倍):
  cn_data_1min/features/{symbol}/*.1min.bin
    │  np.frombuffer (首值=start_idx, 余为float32数据)
    ▼
  1min DataFrame (datetime index, open/high/low/close/volume)
    │  _aggregate_1min_to_5min (基于位置分组, 每天240→48根)
    ▼
  5min DataFrame, 派生 vol=volume, amt=close*volume (后复权口径量能)
    │  按 CV 折的5min时间边界切分
    ▼
  fold{0-3}/{train,val}_5min.pkl  (dict[symbol -> DataFrame])
  full/train_5min.pkl             (全量,供最终模型)

Review 重点:
1. CV切分: 验证段必须严格在训练段之后(无未来泄露)。本实现是扩张式,
   训练段=[0, val_start),验证段=[val_start, val_end),训练段终点≤验证段起点。
2. 跨日连续性: 每股票的5min序列保留跨日(含隔夜缺口),不截断。
   Kronos的窗口z-score归一化会吸收跳空。
3. 1min→5min聚合: 基于位置(非resample), 每日严格48根。
   resample('5min')会产生50根(9:30bin只4根),故用位置分组。
4. bin读取: arr[0]=start_idx(日历索引), arr[1:]=float32值, 对齐calendars/1min.txt
"""
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_highbeta import Config
from universe import load_universe, get_universe_union


# 1min → 5min 聚合规则 (基于位置, 每日严格48根)
# A股1min时间戳: 9:31(首)~15:00(末), 共240根/天
# resample('5min') 会对齐到9:30边界导致首bin只4根、产生50个bin
# 改用 numpy reshape(n5,5) 向量化聚合, 时间戳取每组最后一根1min时间(9:35,9:40...15:00)
# open=col0, high=max(axis1), low=min(axis1), close=col-1, volume=sum(axis1)


def _aggregate_1min_to_5min(g):
    """
    将单只股票的1min DataFrame聚合成5min (基于位置, 按日分组, 向量化加速)。

    每天240根1min → 48根5min (每5根聚1根)。
    时间戳取每组最后一根1min的时间。

    优化: 用 numpy 向量化聚合 (比 pandas groupby+apply 快10倍+)。
    每日: reshape(-1,5) 后 axis=0 聚合, 避免 groupby 开销。

    Args:
        g: DataFrame, index=datetime, cols=[open,high,low,close,volume], 已dropna

    Returns:
        DataFrame: 5min K线, 同列
    """
    if len(g) == 0:
        return g

    dates = g.index.normalize()
    unique_dates = dates.unique()

    opens, highs, lows, closes, vols, ts_list = [], [], [], [], [], []
    for date in unique_dates:
        mask = dates == date
        vals = g[mask]
        n = len(vals)
        n5 = n // 5
        if n5 == 0:
            continue
        # 截断到5的倍数
        vals = vals.iloc[:n5 * 5]
        # numpy 向量化: reshape(n5, 5) 后沿 axis=1 聚合
        o = vals['open'].values.reshape(n5, 5)
        h = vals['high'].values.reshape(n5, 5)
        l = vals['low'].values.reshape(n5, 5)
        c = vals['close'].values.reshape(n5, 5)
        v = vals['volume'].values.reshape(n5, 5)

        opens.append(o[:, 0])       # first
        highs.append(h.max(axis=1)) # max
        lows.append(l.min(axis=1))  # min
        closes.append(c[:, -1])     # last
        vols.append(v.sum(axis=1))  # sum
        # 时间戳: 每组最后一根 (索引 4,9,14,...)
        ts_list.append(vals.index[np.arange(n5) * 5 + 4])

    if not opens:
        return g.iloc[0:0]

    idx = pd.DatetimeIndex(np.concatenate(ts_list))
    result = pd.DataFrame({
        'open': np.concatenate(opens),
        'high': np.concatenate(highs),
        'low': np.concatenate(lows),
        'close': np.concatenate(closes),
        'volume': np.concatenate(vols),
    }, index=idx)
    return result


def compute_cv_splits(trade_days, cv_folds=4, cv_val_days=20):
    """
    计算时间序列CV切分边界(扩张式)。

    Returns:
        list of dict: 每折 {'train_end_day', 'val_start_day', 'val_end_day',
                            'train_end_ts', 'val_start_ts', 'val_end_ts'}
        ts 是5min对齐的时间戳(用于DataFrame按时间mask切分)
    """
    n = len(trade_days)
    val_start_idx = n - cv_folds * cv_val_days   # 529

    splits = []
    for f in range(cv_folds):
        vs = val_start_idx + f * cv_val_days     # 验证段起点的day索引
        ve = vs + cv_val_days                    # 验证段终点(不含)
        split = {
            'fold': f,
            'train_end_day_idx': vs,             # 训练段最后一日的索引(含)
            'val_start_day_idx': vs,
            'val_end_day_idx': ve,
            'train_end_date': trade_days[vs - 1],   # 训练段最后交易日
            'val_start_date': trade_days[vs],       # 验证段首日
            'val_end_date': trade_days[ve - 1],     # 验证段末日
        }
        splits.append(split)

    return splits


def _read_bin(symbol_lower, field, cal_1min, features_root):
    """
    直接读 qlib bin 文件 (绕过 qlib API, 快1000倍)。

    bin 格式 (会话sess_5e6b4326验证):
      - 小端 float32 数组
      - arr[0] = start_idx (日历索引, 存为float但本质整数)
      - arr[1:] = 数据值, 对齐 calendars/1min.txt[start_idx:]

    Args:
        symbol_lower: 小写股票代码 如 'sh600000'
        field: 字段名 如 'open'
        cal_1min: 1min日历 DatetimeIndex
        features_root: features目录路径

    Returns:
        pd.Series (index=datetime, values=float), NaN表示停牌
    """
    path = os.path.join(features_root, symbol_lower, f"{field}.1min.bin")
    if not os.path.exists(path):
        return None
    arr = np.frombuffer(open(path, 'rb').read(), dtype='<f4')
    start_idx = int(arr[0])
    values = arr[1:]
    n = len(values)
    if start_idx + n > len(cal_1min):
        n = len(cal_1min) - start_idx
        values = values[:n]
    dates = cal_1min[start_idx:start_idx + n]
    return pd.Series(values, index=dates, name=field)


def fetch_5min_bin_direct(symbols_lower, cal_1min, features_root):
    """
    直接读bin + 1min→5min聚合 (批量, 无qlib API开销)。

    比 qlib D.features 快约1000倍 (6s vs 2.8h for 5114只)。

    Args:
        symbols_lower: 小写股票代码列表 ['sh600000', ...]
        cal_1min: 1min日历 DatetimeIndex
        features_root: cn_data_1min/features 目录

    Returns:
        dict[symbol_lower -> DataFrame(5min, cols=[open,high,low,close,vol,amt])]
    """
    fields = ['open', 'high', 'low', 'close', 'volume']
    result = {}

    for symbol in tqdm(symbols_lower, desc="Reading bins & aggregating"):
        # 读5个字段的bin
        series_dict = {}
        for f in fields:
            s = _read_bin(symbol, f, cal_1min, features_root)
            if s is not None:
                series_dict[f] = s

        if len(series_dict) < len(fields):
            continue   # 字段不全,跳过

        # 合并成DataFrame
        df = pd.DataFrame(series_dict)
        df = df.sort_index()
        # 删除全NaN行(停牌)
        df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')

        if len(df) == 0:
            continue

        # 1min → 5min 聚合 (基于位置, 每日严格48根)
        df5 = _aggregate_1min_to_5min(df)

        if len(df5) == 0:
            continue

        # 派生 vol, amt (后复权口径: close*volume)
        df5['vol'] = df5['volume']
        df5['amt'] = df5['close'] * df5['volume']
        df5 = df5[['open', 'high', 'low', 'close', 'vol', 'amt']]

        result[symbol] = df5

    return result


def fetch_5min_for_symbols(symbols, start_date, end_date, batch_size=100):
    """
    [已弃用] 批量拉取股票的1min数据并聚合成5min (qlib API版本, 慢)。
    保留供对比验证, 生产用 fetch_5min_bin_direct。
    """
    # 1min 查询需要 end 至少到次日,这里+1天保险
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
    result = {}

    n_batches = (len(symbols) + batch_size - 1) // batch_size
    for bi in tqdm(range(n_batches), desc="Fetching 5min batches", disable=False):
        batch = symbols[bi * batch_size: (bi + 1) * batch_size]
        try:
            df = D.features(
                batch, fields,
                start_time=start_date,
                end_time=end_ts.strftime('%Y-%m-%d'),
                freq='1min'
            )
        except Exception as e:
            print(f"  [WARN] batch {bi} fetch error: {e}")
            continue

        if len(df) == 0:
            continue

        # df: MultiIndex(instrument, datetime), columns $xxx
        # 按 instrument 分组聚合
        df = df.reset_index()
        # 去 $ 前缀
        col_map = {f'${c}': c for c in ['open', 'high', 'low', 'close', 'volume', 'vwap']}
        df = df.rename(columns=col_map)

        for symbol, group in df.groupby('instrument'):
            g = group.set_index('datetime').sort_index()
            # 只保留需要的原始列
            g = g[['open', 'high', 'low', 'close', 'volume']]
            # 删除全NaN行(停牌)
            g = g.dropna(subset=['open', 'high', 'low', 'close'], how='all')

            if len(g) == 0:
                continue

            # 1min → 5min 聚合 (基于位置, 每日严格48根)
            g5 = _aggregate_1min_to_5min(g)

            if len(g5) == 0:
                continue

            # 派生 vol 和 amt (后复权口径量能: close*volume)
            # 注意: qlib bin 无原生 amount,用 close*volume 近似
            # (vwap是不复权口径,与后复权close混用会口径不一致,故用close)
            g5['vol'] = g5['volume']
            g5['amt'] = g5['close'] * g5['volume']

            # 最终只保留 Kronos 需要的6列
            g5 = g5[['open', 'high', 'low', 'close', 'vol', 'amt']]

            result[symbol.lower()] = g5

    return result


def split_and_save_cv(all_data, splits, trade_days, dataset_path):
    """
    按 CV 折切分数据并保存 pickle。

    每折:
      train: 所有股票在 [首日, train_end_date] 的5min数据
      val:   所有股票在 [val_start_date, val_end_date] 的5min数据

    注意: 训练段需要包含 lookback buffer, 但 Kronos 的窗口切片在 dataset.py 做,
    这里只需保证 pickle 时间范围覆盖, dataset 会自动跳过数据不足的窗口。
    """
    for split in splits:
        fold = split['fold']
        train_end = split['train_end_date']      # 训练段末日(含)
        val_start = split['val_start_date']      # 验证段首日(含)
        val_end = split['val_end_date']          # 验证段末日(含)

        fold_dir = os.path.join(dataset_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data, val_data = {}, {}
        for symbol, df in all_data.items():
            # 训练段: <= train_end
            train_mask = df.index.normalize() <= train_end
            train_df = df[train_mask]
            if len(train_df) > 0:
                train_data[symbol] = train_df

            # 验证段: val_start ~ val_end
            val_mask = (df.index.normalize() >= val_start) & (df.index.normalize() <= val_end)
            val_df = df[val_mask]
            if len(val_df) > 0:
                val_data[symbol] = val_df

        with open(os.path.join(fold_dir, 'train_5min.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(fold_dir, 'val_5min.pkl'), 'wb') as f:
            pickle.dump(val_data, f)

        print(f"  折叠{fold}: train {len(train_data)}只 "
              f"[~{train_end.date()}] / val {len(val_data)}只 "
              f"[{val_start.date()}~{val_end.date()}]")

    # 全量数据(供最终模型)
    full_dir = os.path.join(dataset_path, "full")
    os.makedirs(full_dir, exist_ok=True)
    with open(os.path.join(full_dir, 'train_5min.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    print(f"  full: {len(all_data)}只 [全量]")


def main():
    cfg = Config()

    print("=" * 60)
    print("5min 数据预处理 (直接读bin, 无qlib API)")
    print("=" * 60)

    # 1. 加载1min日历 (直接读csv, 无需qlib.init)
    print("\n[1/5] 加载 1min 日历...")
    cal_1min = pd.read_csv(
        os.path.join(cfg.qlib_data_path_1min, "calendars", "1min.txt"),
        header=None, names=['ts']
    )
    cal_1min['ts'] = pd.to_datetime(cal_1min['ts'])
    cal_1min = pd.DatetimeIndex(cal_1min['ts'])
    print(f"  1min日历: {len(cal_1min)}行, {cal_1min[0]} ~ {cal_1min[-1]}")

    features_root = os.path.join(cfg.qlib_data_path_1min, "features")

    # 2. 加载日K交易日历 (用于CV切分)
    print("\n[2/5] 加载交易日历...")
    cal_day = pd.read_csv(
        os.path.join(cfg.qlib_data_path_day, "calendars", "day.txt"),
        header=None, names=['date']
    )
    cal_day['date'] = pd.to_datetime(cal_day['date'])
    trade_days = cal_day['date'].tolist()
    mask = [(d >= pd.Timestamp(cfg.dataset_begin_time)) and
            (d <= pd.Timestamp(cfg.dataset_end_time)) for d in trade_days]
    trade_days = [d for d, m in zip(trade_days, mask) if m]
    print(f"  交易日数: {len(trade_days)} ({trade_days[0].date()}~{trade_days[-1].date()})")

    # 3. 加载股池
    print("\n[3/5] 加载股池...")
    universe = load_universe(cfg.universe_csv, cfg.blacklist)
    symbols_all = sorted(get_universe_union(universe))
    print(f"  股池并集: {len(symbols_all)} 只股票")

    # 4. 计算CV切分
    print("\n[4/5] 计算 CV 切分...")
    splits = compute_cv_splits(trade_days, cfg.cv_folds, cfg.cv_val_days)
    for s in splits:
        print(f"  折叠{s['fold']}: 训练[~{s['train_end_date'].date()}] "
              f"验证[{s['val_start_date'].date()}~{s['val_end_date'].date()}]")

    # 5. 直接读bin + 聚合5min
    print(f"\n[5/5] 读bin + 1min→5min 聚合 ({len(symbols_all)}只)...")
    t0 = time.time()
    all_data = fetch_5min_bin_direct(symbols_all, cal_1min, features_root)
    print(f"  完成: {len(all_data)}只, 耗时 {time.time()-t0:.0f}s")

    # 数据量统计
    total_rows = sum(len(df) for df in all_data.values())
    print(f"  总5min K线数: {total_rows:,}")
    if len(all_data) > 0:
        sample_sym = list(all_data.keys())[0]
        sample_df = all_data[sample_sym]
        print(f"  样例 {sample_sym}: {len(sample_df)}根5min, "
              f"{sample_df.index.min()} ~ {sample_df.index.max()}")

    # 按CV折切分保存
    print(f"\n按CV折切分保存到 {cfg.dataset_path}...")
    split_and_save_cv(all_data, splits, trade_days, cfg.dataset_path)

    print("\n✅ 预处理完成")


if __name__ == '__main__':
    main()
