"""
高贝塔指增 DAILY 预处理:读个股日线 bin + 指数 CSV,对齐成 12 维特征序列,4 折 CV 切分。

与 preprocess_5min.py 的差异:
1. 频率:直接读 day.bin,无 1min→5min 聚合
2. 数据源:个股 qlib day.bin + 指数 index_883926.csv(新)
3. 特征:每只股票的 DataFrame 含 12 列(6 个股 + 6 指数),已对齐交易日
4. 产物:fold{N}/{train,val}_daily.pkl, dict[symbol -> DataFrame[12 cols]]

bin 格式(与 5min 相同):
  - 小端 float32 数组
  - arr[0] = start_idx(日历索引)
  - arr[1:] = 数据值,对齐 calendars/day.txt[start_idx:]

用法:
    cd /home/zxh/projects/Kronos/finetune
    PYTHONPATH=. python enhancement/preprocess_daily.py
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

from enhancement.config_daily import Config
from preprocess_5min import compute_cv_splits  # 复用已验证的扩张式 CV 切分


def load_trade_days(qlib_path: str) -> list[str]:
    """读 calendars/day.txt 交易日历"""
    cal_path = os.path.join(qlib_path, "calendars", "day.txt")
    with open(cal_path) as f:
        days = [line.strip() for line in f if line.strip()]
    return days


def read_day_bin(symbol_lower: str, field: str, cal_day, features_root: str):
    """直接读 day.bin,返回 pd.Series(index=date, values=float)"""
    path = os.path.join(features_root, symbol_lower, f"{field}.day.bin")
    if not os.path.exists(path):
        return None
    arr = np.frombuffer(open(path, "rb").read(), dtype="<f4")
    start_idx = int(arr[0])
    values = arr[1:]
    n = len(values)
    if start_idx + n > len(cal_day):
        n = len(cal_day) - start_idx
        values = values[:n]
    dates = cal_day[start_idx:start_idx + n]
    return pd.Series(values, index=dates, name=field)


def fetch_stock_daily(symbol: str, cal_day, features_root: str,
                      start_date: str, end_date: str) -> pd.DataFrame | None:
    """读单只股票 6 维日线 OHLCV,返回对齐到 [start_date, end_date] 的 DataFrame"""
    symbol_lower = symbol.lower()
    fields_map = {"open": "open", "high": "high", "low": "low",
                  "close": "close", "volume": "volume", "amount": "amount"}
    series_dict = {}
    for kronos_field, qlib_field in fields_map.items():
        s = read_day_bin(symbol_lower, qlib_field, cal_day, features_root)
        if s is not None:
            series_dict[kronos_field] = s
    if "close" not in series_dict:  # close 是必需
        return None

    df = pd.DataFrame(series_dict).sort_index()
    # 派生 kronos 约定的 vol/amt(与 config_highbeta 一致:amt = close*volume)
    if "volume" in df:
        df["vol"] = df["volume"]
    else:
        df["vol"] = 0.0
    if "amount" in df:
        df["amt"] = df["amount"]
    elif "close" in df and "volume" in df:
        df["amt"] = df["close"] * df["volume"]  # 后复权口径
    else:
        df["amt"] = 0.0

    # 统一列顺序为 [open, high, low, close, vol, amt]
    df = df[["open", "high", "low", "close", "vol", "amt"]]

    # 按交易日历对齐(重索引到完整日历,停牌产生 NaN)
    full_idx = cal_day[(cal_day >= start_date) & (cal_day <= end_date)]
    df = df.reindex(full_idx)
    return df


def load_index_daily(index_csv: str, cal_day, start_date: str, end_date: str) -> pd.DataFrame:
    """读指数日线 CSV,返回对齐到交易日历的 DataFrame(6 列 idx_*)"""
    df = pd.read_csv(index_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    # 重命名列
    df = df.rename(columns={
        "open": "idx_open", "high": "idx_high", "low": "idx_low",
        "close": "idx_close", "volume": "idx_vol", "amount": "idx_amt",
    })
    df = df[["idx_open", "idx_high", "idx_low", "idx_close", "idx_vol", "idx_amt"]]
    full_idx = cal_day[(cal_day >= start_date) & (cal_day <= end_date)]
    df = df.reindex(full_idx)
    # 指数前向填充(指数一般不停牌,但保险起见)
    df = df.ffill().bfill()
    return df


def split_and_save_cv(all_data: dict, splits: list, trade_days: list, dataset_path: str,
                      lookback_window: int = 240, predict_window: int = 5):
    """按扩张式 CV 切分,每折保存 train/val pickle。

    关键修正(日频特有):
    - 日频验证段只有 cv_val_days=20 日,远小于 window=246
    - 所以 val pickle 必须包含 [val_start - lookback_buffer, val_end],
      让 Dataset 能切出"lookback 用历史 + predict 落在验证段"的窗口
    - 这不算泄露:lookback 是已知历史,predict 才是要预测的

    train pickle: [首日, train_end_date] 全部(扩张式)
    val pickle:   [val_start - lookback_buffer, val_end](含 lookback 历史)
    """
    os.makedirs(dataset_path, exist_ok=True)

    # val 的 lookback buffer:至少 lookback + predict + 1 个交易日
    buffer = lookback_window + predict_window + 1

    for split in splits:
        fold = split["fold"]
        train_end = split["train_end_date"]
        val_start = split["val_start_date"]
        val_end = split["val_end_date"]

        # val 起点往前推 buffer 个交易日(在 trade_days 里找)
        # trade_days 是完整目标范围,但 val 之前的历史在 train 段里
        # 直接在每只股票的 df 里按位置取末尾 buffer + cv_val_days 根
        # 但要保证 val 段正好是 [val_start, val_end]
        fold_dir = os.path.join(dataset_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data, val_data = {}, {}
        for symbol, df in all_data.items():
            # 训练段: <= train_end
            train_df = df[df.index <= train_end]
            if len(train_df) >= buffer:
                train_data[symbol] = train_df

            # 验证段:包含 lookback buffer,即 [val_start 之前 buffer 根, val_end]
            # 用 trade_days 找 val_start 在完整日历中的位置,往前推 buffer
            val_mask = df.index <= val_end
            val_df_full = df[val_mask]
            # 只保留末尾(buffer + cv_val_days)根,保证有足够 lookback 且 predict 落在验证段
            if len(val_df_full) >= buffer:
                val_data[symbol] = val_df_full.iloc[-(buffer + 20):]  # buffer + 验证段 20 日

        with open(os.path.join(fold_dir, "train_daily.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        with open(os.path.join(fold_dir, "val_daily.pkl"), "wb") as f:
            pickle.dump(val_data, f)

        print(f"  fold{fold}: train {len(train_data)} 只 / val {len(val_data)} 只 "
              f"[val 含 lookback buffer, 末尾 20 日为 {val_start}~{val_end}]")

    # 全量模型(无独立 val,训练用全部数据;val 复用 fold3)
    full_dir = os.path.join(dataset_path, "full")
    os.makedirs(full_dir, exist_ok=True)
    full_data = {s: df for s, df in all_data.items() if len(df) >= buffer}
    with open(os.path.join(full_dir, "train_daily.pkl"), "wb") as f:
        pickle.dump(full_data, f)
    print(f"  full: train {len(full_data)} 只 [全量, val 复用 fold3]")


def main():
    cfg = Config()
    print("=" * 60)
    print("高贝塔指增 DAILY 预处理")
    print("=" * 60)
    print(f"数据源: {cfg.qlib_data_path_day}")
    print(f"股池: {cfg.universe_csv}")
    print(f"指数: {cfg.index_csv}")
    print(f"范围: {cfg.dataset_begin_time} ~ {cfg.dataset_end_time}")
    print()

    # 1. 读交易日历
    # 注意:必须用【完整日历】读 bin(bin 的 start_idx 是对完整日历的索引),
    # 然后在 fetch_stock_daily 里 reindex 到目标范围
    features_root = os.path.join(cfg.qlib_data_path_day, "features")
    trade_days_full = load_trade_days(cfg.qlib_data_path_day)
    cal_day_full = pd.DatetimeIndex(trade_days_full).strftime("%Y-%m-%d")
    # 目标范围(用于最后截取)
    trade_days = [d for d in trade_days_full
                  if cfg.dataset_begin_time <= d <= cfg.dataset_end_time]
    print(f"完整交易日历: {len(trade_days_full)} 日")
    print(f"目标范围交易日: {len(trade_days)} ({trade_days[0]} ~ {trade_days[-1]})")
    cal_day = cal_day_full  # read_day_bin 用完整日历

    # 2. 读成分股并集
    uni = pd.read_csv(cfg.universe_csv)
    symbols = sorted(uni["code_qlib"].unique())
    print(f"股池并集: {len(symbols)} 只")

    # 3. 读指数日线(全局共享,对齐到交易日)
    print(f"\n加载指数 {cfg.index_symbol}...")
    idx_df = load_index_daily(cfg.index_csv, cal_day,
                              cfg.dataset_begin_time, cfg.dataset_end_time)
    print(f"  指数日线: {len(idx_df)} 行, NaN 数: {idx_df.isna().sum().sum()}")
    if idx_df.isna().sum().sum() > 0:
        print("  ⚠️ 指数有 NaN,请检查 index_csv")

    # 4. 逐股票读日线 + 拼指数特征
    print(f"\n读 {len(symbols)} 只股票日线 + 拼接指数特征...")
    all_data = {}
    missing = 0
    for symbol in tqdm(symbols, desc="Reading day bins"):
        stock_df = fetch_stock_daily(
            symbol, cal_day, features_root,
            cfg.dataset_begin_time, cfg.dataset_end_time,
        )
        if stock_df is None or len(stock_df) < 246:
            missing += 1
            continue
        # 拼接指数特征(同一时间轴)
        combined = pd.concat([stock_df, idx_df], axis=1)
        # 删除个股停牌日(close 为 NaN)——这些天无信号意义
        combined = combined.dropna(subset=["close"])
        if len(combined) >= 246:
            all_data[symbol] = combined

    print(f"\n有效股票: {len(all_data)}/{len(symbols)} (跳过 {missing} 只数据不足)")

    # 5. OHLC 一致性快速检查(个股)
    violations = 0
    for s, df in all_data.items():
        bad_h = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
        bad_l = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
        violations += bad_h + bad_l
    print(f"个股 OHLC 违规总行数: {violations}")

    # 6. CV 切分 + 保存
    print(f"\nCV 切分({cfg.cv_folds} 折扩张式,验证 {cfg.cv_val_days} 日/折)...")
    splits = compute_cv_splits(trade_days, cfg.cv_folds, cfg.cv_val_days)
    for sp in splits:
        print(f"  fold{sp['fold']}: train~{sp['train_end_date']} | "
              f"val={sp['val_start_date']} ~ {sp['val_end_date']}")

    print(f"\n保存到 {cfg.dataset_path}...")
    split_and_save_cv(all_data, splits, trade_days, cfg.dataset_path,
                      lookback_window=cfg.lookback_window,
                      predict_window=cfg.predict_window)

    # 保存切分边界(回测要用)
    import json
    splits_path = os.path.join(cfg.dataset_path, "cv_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2, default=str)
    print(f"\nCV 边界: {splits_path}")

    print("\n✅ 预处理完成。下一步:")
    print(f"  KRONOS_CONFIG=config_daily KRONOS_FOLD=0 python smoke_test.py")
    print(f"  或 bash enhancement/run_daily.sh train-all")


if __name__ == "__main__":
    main()
