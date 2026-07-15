"""
5min 数据验证脚本。

用法:
  # 验证某个 fold 的 pickle
  python check_data.py --fold 0
  # 验证 full
  python check_data.py --fold full
  # 抽样画图(需要 matplotlib)
  python check_data.py --fold 0 --plot sh600000
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_highbeta import Config


def load_fold(fold, data_type='train'):
    """加载某折的 pickle。"""
    cfg = Config()
    path = os.path.join(cfg.get_cv_fold_dir(fold), f"{data_type}_5min.pkl")
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_basic(data, label):
    """基本统计检查。"""
    print(f"\n{'='*60}")
    print(f"[{label}] 基本检查")
    print(f"{'='*60}")
    print(f"股票数: {len(data)}")

    if len(data) == 0:
        print("❌ 无数据")
        return

    # 每股票5min根数
    counts = [len(df) for df in data.values()]
    print(f"每股票5min根数: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.0f} total={sum(counts):,}")

    # 时间范围
    all_starts = [df.index.min() for df in data.values()]
    all_ends = [df.index.max() for df in data.values()]
    print(f"时间范围: {min(all_starts)} ~ {max(all_ends)}")

    # 列检查
    sample_sym = list(data.keys())[0]
    sample_df = data[sample_sym]
    expected_cols = ['open', 'high', 'low', 'close', 'vol', 'amt']
    print(f"列: {sample_df.columns.tolist()} (期望: {expected_cols})")
    if list(sample_df.columns) != expected_cols:
        print(f"❌ 列不匹配!")

    # OHLC 合理性: high>=low, high>=max(open,close), low<=min(open,close)
    violations = 0
    for sym, df in list(data.items())[:50]:  # 抽查50只
        bad = ((df['high'] < df['low']) |
               (df['high'] < df[['open', 'close']].max(axis=1) - 1e-4) |
               (df['low'] > df[['open', 'close']].min(axis=1) + 1e-4))
        violations += bad.sum()
    print(f"OHLC一致性违规(抽查50只): {violations} {'✅' if violations == 0 else '⚠️'}")


def check_daily_bars(data, label):
    """检查每天的5min根数是否为48。"""
    print(f"\n[{label}] 每日5min根数检查")
    # 抽查若干只股票
    for sym in list(data.keys())[:3]:
        df = data[sym]
        daily_counts = df.groupby(df.index.normalize()).size()
        non48 = daily_counts[daily_counts != 48]
        if len(non48) > 0:
            print(f"  {sym}: {len(non48)}/{len(daily_counts)} 天根数≠48")
            print(f"    异常值: {non48.head().to_dict()}")
        else:
            print(f"  {sym}: 全部{len(daily_counts)}天均为48根 ✅")


def check_cross_day(data, label):
    """检查跨日连续性(序列是否保留跨日,无截断)。"""
    print(f"\n[{label}] 跨日连续性检查")
    for sym in list(data.keys())[:2]:
        df = data[sym]
        dates = df.index.normalize().unique()
        if len(dates) < 2:
            continue
        # 检查相邻两天的5min序列是否连续(日期相邻)
        d1, d2 = dates[0], dates[1]
        day1_end = df[df.index.normalize() == d1].index[-1]
        day2_start = df[df.index.normalize() == d2].index[0]
        gap = day2_start - day1_end
        print(f"  {sym}: {d1.date()}末={day1_end.strftime('%H:%M')} → "
              f"{d2.date()}首={day2_start.strftime('%H:%M')} (隔夜gap={gap})")
        # 检查是否所有日期连续(无缺失交易日)
        date_gaps = []
        for i in range(1, min(len(dates), 20)):
            if (dates[i] - dates[i-1]).days > 10:  # 超10天可能是停牌
                date_gaps.append((dates[i-1].date(), dates[i].date()))
        if date_gaps:
            print(f"    ⚠️ 长停牌: {date_gaps[:3]}")


def check_window_count(data, lookback=240, predict=48):
    """统计可用的训练窗口数。"""
    window = lookback + predict + 1
    total_windows = 0
    for sym, df in data.items():
        n_windows = max(0, len(df) - window + 1)
        total_windows += n_windows
    print(f"\n可用窗口数 (window={window}): {total_windows:,}")
    return total_windows


def plot_sample(data, symbol):
    """画某只股票的5min收盘价。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装,跳过画图")
        return

    if symbol not in data:
        print(f"❌ {symbol} 不在数据中")
        return
    df = data[symbol]
    # 取最近5个交易日
    recent_dates = sorted(df.index.normalize().unique())[-5:]
    mask = df.index.normalize().isin(recent_dates)
    plot_df = df[mask]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(plot_df.index, plot_df['close'], 'b-', linewidth=0.8)
    axes[0].set_title(f'{symbol} 5min close (recent 5 days)')
    axes[0].set_ylabel('close (后复权)')
    axes[1].bar(plot_df.index, plot_df['vol'], width=0.003, color='orange')
    axes[1].set_title('volume')
    axes[1].set_ylabel('vol')

    plt.tight_layout()
    out = f'check_{symbol}_5min.png'
    plt.savefig(out, dpi=100)
    print(f"✅ 图已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default='0', help='fold 编号 (0-3 或 full)')
    parser.add_argument('--plot', default=None, help='画某股票图, 如 sh600000')
    args = parser.parse_args()

    cfg = Config()
    print(f"配置: lookback={cfg.lookback_window} predict={cfg.predict_window} "
          f"window={cfg.lookback_window + cfg.predict_window + 1}")

    # 检查 train
    data_train = load_fold(args.fold, 'train')
    check_basic(data_train, f"fold{args.fold}/train")
    check_daily_bars(data_train, f"fold{args.fold}/train")
    check_cross_day(data_train, f"fold{args.fold}/train")
    check_window_count(data_train, cfg.lookback_window, cfg.predict_window)

    # 检查 val (full 没有 val)
    val_path = os.path.join(cfg.get_cv_fold_dir(args.fold), "val_5min.pkl")
    if os.path.exists(val_path):
        data_val = load_fold(args.fold, 'val')
        check_basic(data_val, f"fold{args.fold}/val")
        check_daily_bars(data_val, f"fold{args.fold}/val")
        check_window_count(data_val, cfg.lookback_window, cfg.predict_window)

    # 画图
    if args.plot:
        plot_sample(data_train, args.plot)


if __name__ == '__main__':
    main()
