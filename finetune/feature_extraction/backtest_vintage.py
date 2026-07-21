"""
Vintage 重叠回测(实验 5 用)。

为什么需要 vintage:
  qlib 的 hold_thresh 只管卖不管买,做不了真正 T+5 持有。
  vintage 模式:每天 T 都形成一个 top-K 组合(V_T),持有 hold_days 天。
  基金每日收益 = 当前在跑的所有 vintage 的等权平均。
  产出真日收益序列,Sharpe 与 T+1 可比。

用 qlib.contrib.evaluate.risk_analysis 算指标,N=238(qlib 约定,非 252)。

用法:
    cd /home/zxh/projects/3.qlib_ifind_beta
    python /home/zxh/projects/Kronos/finetune/feature_extraction/backtest_vintage.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import qlib
qlib.init(provider_uri='/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root', region='cn')
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis


def read_bin(sym_lower, field):
    p = f'/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/features/{sym_lower}/{field}.day.bin'
    if not os.path.exists(p):
        return None
    arr = np.fromfile(p, dtype='<f4')
    if arr.size == 0:
        return None
    return int(arr[0]), arr[1:]


def load_daily_returns():
    """读所有股票的日收益(close 的 pct_change),返回 dict[sym -> Series(date→ret)]"""
    with open('/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/calendars/day.txt') as f:
        days = [l.strip() for l in f if l.strip()]
    days_ts = pd.to_datetime(days)
    return days, days_ts


def load_benchmark_daily_ret():
    """SH000300 日收益"""
    r = read_bin('sh000300', 'close')
    if r is None:
        return None
    si, close = r
    with open('/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/calendars/day.txt') as f:
        days = [l.strip() for l in f if l.strip()]
    days_ts = pd.to_datetime(days)
    s = pd.Series(close, index=days_ts[si:si+len(close)])
    return s.pct_change(fill_method=None)


def backtest_vintage(exp_name, hold_days=5, topk=10, cost_one_side=0.001):
    """vintage 重叠回测。

    Args:
        exp_name: qlib experiment 名
        hold_days: 每个 vintage 持有天数(T+1 策略用 1,T+5 用 5)
        topk: 选股数
        cost_one_side: 单边成本(open 0.0005 + close 0.0015 简化为 0.001)

    Returns:
        dict: IC, RankIC, 年化超额, 夏普, 最大回撤, 胜率, 日均超额 bps
    """
    exp = R.get_exp(experiment_name=exp_name)
    rec = list(exp.list_recorders().values())[0]
    pred = rec.load_object('pred.pkl')
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    # 加载日历和基准
    with open('/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/calendars/day.txt') as f:
        days = [l.strip() for l in f if l.strip()]
    days_ts = pd.to_datetime(days)
    day_to_idx = {d: i for i, d in enumerate(days_ts)}
    bench_ret = load_benchmark_daily_ret()

    # 加载所有涉及股票的日收益
    print(f"  [{exp_name}] 加载股票日收益...")
    syms = pred.index.get_level_values('instrument').unique()
    stock_ret = {}  # sym -> ndarray(daily ret, len=总交易日),按 day_idx 对齐
    for sym in syms:
        r = read_bin(sym.lower(), 'close')
        if r is None:
            continue
        si, close = r
        if len(close) < 2:
            continue
        rets = np.full(len(days_ts), np.nan)
        # close[0] 对应 days[si],ret[i] = close[i+1]/close[i]-1
        valid_end = min(len(close), len(days_ts) - si)
        for i in range(valid_end - 1):
            if close[i] > 0 and not np.isnan(close[i+1]) and not np.isnan(close[i]):
                rets[si + i + 1] = close[i+1] / close[i] - 1
        stock_ret[sym] = rets

    # pred 按 date 分组
    pred_by_date = {}
    for (dt, sym), score in pred.items():
        if sym not in stock_ret:
            continue
        d = pd.Timestamp(dt)
        if d not in day_to_idx:
            continue
        pred_by_date.setdefault(d, []).append((sym, score))

    # vintage 回测
    # 每个 vintage V_T:在 day_idx=T 用 pred 选 topk,持有 hold_days 天
    # V_T 在 day_idx=T+1..T+hold_days 的日收益 = topk 股票日收益均值
    # 基金日收益 = 当日所有活跃 vintage 的等权均值
    test_dates = sorted(pred_by_date.keys())
    daily_port = {}  # day_idx → list of vintage returns active that day
    daily_turnover_cost = {}

    for T_date in test_dates:
        T_idx = day_to_idx[T_date]
        # 选 topk
        candidates = pred_by_date[T_date]
        candidates.sort(key=lambda x: -x[1])
        top = [s for s, _ in candidates[:topk]]
        if len(top) < topk:
            continue
        # 这个 vintage 在 T+1..T+hold_days 活跃
        for h in range(1, hold_days + 1):
            active_idx = T_idx + h
            if active_idx >= len(days_ts):
                break
            # vintage 在 active_idx 这一天的收益
            rets = []
            for sym in top:
                r = stock_ret[sym][active_idx] if sym in stock_ret else np.nan
                if not np.isnan(r):
                    rets.append(r)
            if rets:
                daily_port.setdefault(active_idx, []).append(np.mean(rets))
        # 换手成本:这个 vintage 在 T+1 建立(买),在 T+hold_days+1 清算(卖)
        # 简化:在 T+1 这天扣 round trip 成本的 1/hold_days(每天分摊)
        # 实际:round trip = 2 * cost_one_side,每天 1/hold_days 的 vintage 换
        # → 日成本 = (1/hold_days) * 2 * cost_one_side
        daily_cost = (1.0 / hold_days) * 2 * cost_one_side

    # 算每日组合收益(活跃 vintage 等权)
    port_ret_list = []
    bench_ret_list = []
    cost_list = []
    for idx in sorted(daily_port.keys()):
        d = days_ts[idx]
        port_mean = np.mean(daily_port[idx])
        b = bench_ret.get(d, np.nan) if bench_ret is not None else np.nan
        port_ret_list.append(port_mean)
        bench_ret_list.append(b)
        cost_list.append((1.0 / hold_days) * 2 * cost_one_side)

    port_ret_s = pd.Series(port_ret_list)
    bench_ret_s = pd.Series(bench_ret_list)
    cost_s = pd.Series(cost_list)

    # 超额(扣成本)
    excess_net = port_ret_s - bench_ret_s - cost_s
    excess_gross = port_ret_s - bench_ret_s

    # IC/RankIC(逐日)
    ic_list, ric_list = [], []
    for T_date in test_dates:
        if T_date not in pred_by_date:
            continue
        T_idx = day_to_idx[T_date]
        cands = pred_by_date[T_date]
        # 实际 horizon 收益(用 close[T+hold]/close[T]-1 近似,与 label 口径略不同)
        rets = []
        for sym, score in cands:
            if sym not in stock_ret:
                continue
            # T 到 T+hold_days 的累计收益
            r_arr = stock_ret[sym]
            seg = r_arr[T_idx+1:T_idx+hold_days+1]
            if np.isnan(seg).all() or len(seg) < hold_days:
                continue
            cum = np.nanprod(1 + seg) - 1
            rets.append((score, cum))
        if len(rets) < 20:
            continue
        scores = [x[0] for x in rets]
        actuals = [x[1] for x in rets]
        ic = np.corrcoef(scores, actuals)[0, 1]
        r_ic, _ = spearmanr(scores, actuals)
        if not np.isnan(ic):
            ic_list.append(ic)
        if not np.isnan(r_ic):
            ric_list.append(r_ic)

    # 用 qlib risk_analysis(模式 sum,N=238)
    if len(excess_net) == 0:
        return None
    risk = risk_analysis(excess_net.dropna(), freq='day')
    risk_gross = risk_analysis(excess_gross.dropna(), freq='day')
    risk_port = risk_analysis(port_ret_s.dropna(), freq='day')

    return {
        'exp': exp_name,
        'n_days': len(excess_net),
        'hold_days': hold_days,
        'IC': np.mean(ic_list) if ic_list else np.nan,
        'RankIC': np.mean(ric_list) if ric_list else np.nan,
        'ann_excess_net': float(risk.loc['annualized_return', 'risk']),
        'sharpe_net': float(risk.loc['information_ratio', 'risk']),
        'max_drawdown': float(risk.loc['max_drawdown', 'risk']),
        'ann_excess_gross': float(risk_gross.loc['annualized_return', 'risk']),
        'ann_port_return': float(risk_port.loc['annualized_return', 'risk']),
        'mean_daily_excess_bps': float(excess_gross.mean() * 10000),
        'win_rate': float((excess_net > 0).mean()),
    }


def main():
    # 四跑全用 vintage 回测(包括 T+1 的,保证可比)
    # T+1 的 hold_days=1,T+5 的 hold_days=5
    runs = [
        ('A baseline (T+1)',  'kronos_ablation_A_baseline',     1),
        ('D 精选 Kronos (T+1)','kronos_ablation_D_selected',     1),
        ('A_T5 baseline (T+5)','kronos_ablation_E_T5_baseline',  5),
        ('E_T5 +Kronos (T+5)', 'kronos_ablation_E_T5_selected',  5),
    ]

    print("=" * 100)
    print("Vintage 重叠回测(qlib risk_analysis,N=238)")
    print("=" * 100)

    results = []
    for label, exp_name, hold in runs:
        print(f"\n--- {label} (hold={hold}d) ---")
        try:
            r = backtest_vintage(exp_name, hold_days=hold, topk=10, cost_one_side=0.001)
            if r:
                r['label'] = label
                results.append(r)
                print(f"  IC={r['IC']:+.4f}  RankIC={r['RankIC']:+.4f}")
                print(f"  年化超额(扣成本)={r['ann_excess_net']*100:+.2f}%  夏普={r['sharpe_net']:+.2f}  回撤={r['max_drawdown']*100:.2f}%")
                print(f"  年化超额(毛)   ={r['ann_excess_gross']*100:+.2f}%")
                print(f"  胜率={r['win_rate']*100:.1f}%  日均超额={r['mean_daily_excess_bps']:.1f}bps  天数={r['n_days']}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    # 汇总
    if results:
        print("\n" + "=" * 100)
        print("汇总")
        print("=" * 100)
        print(f"{'跑':<28s} {'hold':>5s} {'IC':>8s} {'RankIC':>8s} {'年化超额(净)':>13s} {'夏普':>6s} {'回撤':>8s} {'胜率':>6s}")
        print("-" * 100)
        for r in results:
            print(f"{r['label']:<28s} {r['hold_days']:>5d} {r['IC']:>8.4f} {r['RankIC']:>8.4f} "
                  f"{r['ann_excess_net']*100:>+12.2f}% {r['sharpe_net']:>6.2f} "
                  f"{r['max_drawdown']*100:>7.2f}% {r['win_rate']*100:>5.1f}%")

        # 关键对比
        if len(results) >= 4:
            a_t1 = next(r for r in results if 'A baseline' in r['label'])
            e_t5 = next(r for r in results if 'E_T5' in r['label'])
            a_t5 = next(r for r in results if 'A_T5' in r['label'])
            print(f"\n=== 关键判定 ===")
            print(f"  A baseline (T+1):    超额 {a_t1['ann_excess_net']*100:+.2f}%  夏普 {a_t1['sharpe_net']:.2f}")
            print(f"  E_T5 +Kronos (T+5):  超额 {e_t5['ann_excess_net']*100:+.2f}%  夏普 {e_t5['sharpe_net']:.2f}")
            print(f"  A_T5 baseline (T+5): 超额 {a_t5['ann_excess_net']*100:+.2f}%  夏普 {a_t5['sharpe_net']:.2f}")
            print(f"\n  E_T5 vs A baseline(T+1): 超额 {e_t5['ann_excess_net']-a_t1['ann_excess_net']:+.4f}, 夏普 {e_t5['sharpe_net']-a_t1['sharpe_net']:+.2f}")
            print(f"  E_T5 vs A_T5(同 horizon): 超额 {e_t5['ann_excess_net']-a_t5['ann_excess_net']:+.4f}, 夏普 {e_t5['sharpe_net']-a_t5['sharpe_net']:+.2f}")
            # 判定
            print(f"\n  判定:")
            if e_t5['ann_excess_net'] > a_t1['ann_excess_net'] and e_t5['sharpe_net'] > a_t1['sharpe_net']:
                print(f"    ✅ E_T5 超额 AND 夏普 同时优于 A baseline(T+1)→ Kronos 在中期有效!")
            elif e_t5['ann_excess_net'] > a_t5['ann_excess_net'] and e_t5['sharpe_net'] > a_t5['sharpe_net']:
                print(f"    🟡 E_T5 在同 horizon 下优于 baseline,但整体不如 T+1 baseline → Kronos 改善中期但非突破")
            else:
                print(f"    ❌ E_T5 未同时超过 baseline → Kronos 在中期也无实际 alpha")


if __name__ == '__main__':
    main()
