"""
Kronos ablation 四跑的简单回测(PortAnaRecord 跑不通,自己算)。

策略逻辑(与 qrun yaml 一致):
  - 每个 T 日,取 pred 分数最高的 top-10 股票(等权)
  - 用 label[T] = Ref($close,-1)/$price_941 - 1 作为实际收益
    (label 就是 T 日 9:41 买入、T+1 收盘卖出的真实收益)
  - 超额收益 = 组合收益 - 基准(SH000300)同期收益
  - 成本:单边 0.1%(open_cost + close_cost 各自,简化为 round trip 0.2%)

注意:label.pkl 已经是真实收益(含 price_941 和 T+1 close),
      不需要重新算,直接用 pred 选 top-10、对 label 求均值即可。

用法:
    cd /home/zxh/projects/3.qlib_ifind_beta
    python /home/zxh/projects/Kronos/finetune/feature_extraction/backtest_ablation.py
"""
import os
import sys
import numpy as np
import pandas as pd

# qlib 初始化
import qlib
qlib.init(provider_uri='/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root', region='cn')
from qlib.workflow import R

# 基准指数(SH000300)日收益
def load_benchmark_returns():
    """读 SH000300 日线,算日收益"""
    import numpy as np
    path = '/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/features/sh000300/close.day.bin'
    if not os.path.exists(path):
        return None
    with open('/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/calendars/day.txt') as f:
        days = [l.strip() for l in f if l.strip()]
    arr = np.fromfile(path, dtype='<f4')
    start_idx = int(arr[0])
    close = arr[1:]
    # 对齐到日历
    dates = pd.to_datetime(days[start_idx:start_idx+len(close)])
    ret = pd.Series(close, index=dates).pct_change()
    return ret


def backtest_experiment(exp_name, topk=10, cost_bps=20):
    """对某个 experiment 的第一个 recorder 做简单 topk 回测。

    Args:
        exp_name: qlib experiment 名
        topk: 选股数
        cost_bps: 单边成本(bps),默认 20 = 0.2%(open 0.05% + close 0.15% 简化)

    Returns:
        dict: IC, RankIC, 年化超额, 夏普, 最大回撤, 胜率
    """
    exp = R.get_exp(experiment_name=exp_name)
    recorders = exp.list_recorders()
    rid = list(recorders.keys())[0]
    rec = recorders[rid]

    pred = rec.load_object('pred.pkl')
    label = rec.load_object('label.pkl')

    # pred 和 label 都是 Series,MultiIndex (datetime, instrument)
    # 对齐
    df = pd.DataFrame({'pred': pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred,
                       'label': label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label})
    df = df.dropna(subset=['pred', 'label'])

    # 基准收益
    bench_ret = load_benchmark_returns()
    if bench_ret is None:
        return None

    # IC / RankIC(逐日)
    from scipy.stats import spearmanr
    ic_list, ric_list = [], []
    daily_dates = []
    daily_port_ret = []
    daily_bench_ret = []
    daily_excess = []

    for date, group in df.groupby(level='datetime'):
        if len(group) < topk:
            continue
        # IC
        ic = group['pred'].corr(group['label'])
        ric, _ = spearmanr(group['pred'], group['label'])
        if not np.isnan(ic):
            ic_list.append(ic)
        if not np.isnan(ric):
            ric_list.append(ric)

        # topk 组合:取 pred 最高的 topk 只,等权
        top = group.nlargest(topk, 'pred')
        port_ret = top['label'].mean()  # 组合当日收益(label 已是 T→T+1 收益)

        # 基准收益(同日)
        date_ts = pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date
        b_ret = bench_ret.get(date_ts, np.nan)

        daily_dates.append(date_ts)
        daily_port_ret.append(port_ret)
        daily_bench_ret.append(b_ret)
        if not np.isnan(b_ret):
            daily_excess.append(port_ret - b_ret)
        else:
            daily_excess.append(np.nan)

    # 成本扣除(换手近似:每天换 80% 仓位,round trip 成本)
    cost_per_day = cost_bps / 10000.0  # 单边
    daily_cost = 0.8 * 2 * cost_per_day  # 假设 80% 换手,双边

    excess_s = pd.Series(daily_excess, index=daily_dates).dropna()
    excess_s_net = excess_s - daily_cost  # 扣成本

    # 统计
    n_days = len(excess_s)
    mean_excess = excess_s_net.mean()
    std_excess = excess_s_net.std()
    ann_excess = mean_excess * 252  # 年化(简化,未复利)
    sharpe = mean_excess / std_excess * np.sqrt(252) if std_excess > 0 else 0

    # 累计超额 + 最大回撤
    cum = (1 + excess_s_net).cumprod()
    peak = cum.expanding().max()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # 胜率
    win_rate = (excess_s_net > 0).mean()

    return {
        'exp': exp_name,
        'n_days': n_days,
        'IC': np.mean(ic_list),
        'RankIC': np.mean(ric_list),
        'ann_excess_net': ann_excess,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'mean_daily_excess_bps': mean_excess * 10000,
    }


def main():
    print("=" * 90)
    print("Kronos ablation 四跑回测(test 段 2026-04~07,topk=10)")
    print("=" * 90)

    experiments = [
        ('A baseline (18)', 'kronos_ablation_A_baseline'),
        ('B +base (26)', 'kronos_ablation_B_base'),
        ('C +finetune (26)', 'kronos_ablation_C_finetune'),
        ('D +selected+Wins (21)', 'kronos_ablation_D_selected'),
        ('Champion (18, n_drop=8)', 'minute_enhanced_tk10_nd8'),
    ]

    results = []
    for label, exp_name in experiments:
        try:
            r = backtest_experiment(exp_name)
            if r:
                r['label'] = label
                results.append(r)
                print(f"\n{label} ({exp_name}):")
                print(f"  IC={r['IC']:.4f}  RankIC={r['RankIC']:.4f}")
                print(f"  年化超额(扣成本)= {r['ann_excess_net']*100:+.2f}%")
                print(f"  夏普={r['sharpe']:.2f}  最大回撤={r['max_drawdown']*100:.2f}%  胜率={r['win_rate']*100:.1f}%")
                print(f"  日均超额={r['mean_daily_excess_bps']:.1f}bps  天数={r['n_days']}")
        except Exception as e:
            print(f"\n{label} ({exp_name}): ERROR {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总表
    if results:
        print("\n" + "=" * 90)
        print("汇总")
        print("=" * 90)
        print(f"{'跑':<28s} {'IC':>8s} {'RankIC':>8s} {'年化超额':>10s} {'夏普':>6s} {'回撤':>8s} {'胜率':>6s}")
        print("-" * 90)
        for r in results:
            print(f"{r['label']:<28s} {r['IC']:>8.4f} {r['RankIC']:>8.4f} "
                  f"{r['ann_excess_net']*100:>+9.2f}% {r['sharpe']:>6.2f} "
                  f"{r['max_drawdown']*100:>7.2f}% {r['win_rate']*100:>5.1f}%")

        # A vs C vs D 增量
        if len(results) >= 4:
            a = next(r for r in results if 'A ' in r['label'])
            c = next(r for r in results if 'C ' in r['label'])
            d = next(r for r in results if 'D ' in r['label'])
            print(f"\n增量(vs A baseline):")
            print(f"  C 全 Kronos:    年化超额 {c['ann_excess_net']*100 - a['ann_excess_net']*100:+.2f}%, 夏普 {c['sharpe']-a['sharpe']:+.2f}")
            print(f"  D 精选+Winsor:  年化超额 {d['ann_excess_net']*100 - a['ann_excess_net']*100:+.2f}%, 夏普 {d['sharpe']-a['sharpe']:+.2f}")


if __name__ == '__main__':
    main()
