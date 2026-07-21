"""
止损门:baseline 18 因子(champion)的多 horizon RankIC 衰减诊断。

决定是否继续实验 5(中期反转策略)。
- 若 T+5 RankIC > 0.03 → baseline 在中期仍有 alpha,继续
- 若 T+5 RankIC < 0.01 → baseline 中期已失效,止损

用法:
    cd /home/zxh/projects/3.qlib_ifind_beta
    python /home/zxh/projects/Kronos/finetune/feature_extraction/diag_horizon_decay.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import qlib
qlib.init(provider_uri='/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root', region='cn')
from qlib.workflow import R


def read_bin(sym_lower, field):
    p = f'/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/features/{sym_lower}/{field}.day.bin'
    if not os.path.exists(p):
        return None
    arr = np.fromfile(p, dtype='<f4')
    if arr.size == 0:
        return None
    return int(arr[0]), arr[1:]


def main():
    # champion 的预测(baseline 18 因子)
    exp = R.get_exp(experiment_name='minute_enhanced_tk10_nd8')
    rec = list(exp.list_recorders().values())[0]
    pred = rec.load_object('pred.pkl')
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    pred_df = pred.reset_index()
    pred_df.columns = ['datetime', 'instrument', 'score']
    pred_df['date_str'] = pred_df['datetime'].dt.strftime('%Y-%m-%d')
    print(f"champion pred 样本: {len(pred_df)}")
    print(f"日期范围: {pred_df['date_str'].min()} ~ {pred_df['date_str'].max()}")

    with open('/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/calendars/day.txt') as f:
        days = [l.strip() for l in f if l.strip()]

    # 算多 horizon 的 label = close[T+h] / price_941[T] - 1
    horizons = [1, 2, 3, 5, 10]
    # 缓存:每只股票的 close 和 price_941
    close_cache = {}
    p941_cache = {}
    sym_list = pred_df['instrument'].unique()
    print(f"加载 {len(sym_list)} 只股票的 close + price_941...")
    for sym in sym_list:
        rc = read_bin(sym.lower(), 'close')
        rp = read_bin(sym.lower(), 'price_941')
        if rc is not None:
            close_cache[sym] = rc
        if rp is not None:
            p941_cache[sym] = rp
    print(f"  close: {len(close_cache)}, price_941: {len(p941_cache)}")

    # 每个 horizon 算 RankIC
    print("\n=== baseline 18 因子的多 horizon RankIC 衰减 ===")
    print(f"{'horizon':>8s} {'RankIC':>10s} {'ICIR':>8s} {'n_days':>8s} {'vs T+1':>10s}")
    print("-" * 50)
    results = {}
    for h in horizons:
        records = []
        for _, row in pred_df.iterrows():
            sym = row['instrument']
            date_str = row['date_str']
            if date_str not in days:
                continue
            if sym not in close_cache or sym not in p941_cache:
                continue
            di = days.index(date_str)
            sci, close = close_cache[sym]
            spi, p941 = p941_cache[sym]
            pos_c = di - sci
            pos_p = di - spi
            if pos_c < 0 or pos_c + h >= len(close):
                continue
            if pos_p < 0 or pos_p >= len(p941):
                continue
            c_h = close[pos_c + h]
            c_0 = close[pos_c]  # 不用,但保留
            p0 = p941[pos_p]
            if np.isnan(c_h) or np.isnan(p0) or p0 <= 0:
                continue
            # label[T, h] = close[T+h] / price_941[T] - 1
            records.append((row['datetime'], sym, c_h / p0 - 1, row['score']))
        if not records:
            continue
        rdf = pd.DataFrame(records, columns=['date', 'sym', 'ret', 'pred']).set_index(['date', 'sym'])
        ics = []
        for d, g in rdf.groupby(level='date'):
            if len(g) < 20:
                continue
            r, _ = spearmanr(g['pred'], g['ret'])
            if not np.isnan(r):
                ics.append(r)
        if not ics:
            continue
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / (std_ic + 1e-9)
        results[h] = (mean_ic, icir, len(ics))

    base_ic = results.get(1, (0, 0, 0))[0]
    for h in horizons:
        if h in results:
            m, ir, n = results[h]
            pct = (m / base_ic * 100) if base_ic != 0 else 0
            print(f"T+{h:<7d} {m:>+10.4f} {ir:>+8.3f} {n:>8d} {pct:>+9.0f}%")

    # 止损判定
    print("\n" + "=" * 50)
    print("止损门判定:")
    if 5 in results:
        t5_ic = results[5][0]
        if t5_ic > 0.03:
            print(f"  ✅ T+5 RankIC = {t5_ic:+.4f} > 0.03 → baseline 中期仍有 alpha,继续实验 5")
        elif t5_ic < 0.01:
            print(f"  ❌ T+5 RankIC = {t5_ic:+.4f} < 0.01 → baseline 中期已失效,止损!")
            print(f"     方向不可行:baseline 自己都没 alpha,Kronos 即使中期强也没对照意义")
        else:
            print(f"  ⚠️ T+5 RankIC = {t5_ic:+.4f} 介于 0.01-0.03,边界情况,谨慎推进")
    else:
        print("  ❌ 无法计算 T+5 RankIC(数据不足)")


if __name__ == '__main__':
    main()
