"""
汇总4折回测结果,生成超额收益分析报告。

用法: (4折回测全部完成后)
  python summarize_backtest.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_highbeta import Config


def load_backtest_result(fold):
    cfg = Config()
    path = f"{cfg.save_path}/backtest/backtest_fold{fold}_modelfull.pkl"
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    cfg = Config()
    print("=" * 70)
    print("4折回测结果汇总")
    print("=" * 70)

    all_results = {}
    for fold in range(4):
        r = load_backtest_result(fold)
        if r is None:
            print(f"\nfold{fold}: ❌ 结果不存在")
            continue
        all_results[fold] = r
        m = r['metrics']
        print(f"\nfold{fold} (验证段 {r['daily_details'][0]['date']}~{r['daily_details'][-1]['date']}):")
        print(f"  累计收益: {m['total_return']*100:>8.2f}%")
        print(f"  日均收益: {m['avg_daily_return']*100:>8.3f}%")
        print(f"  夏普比率: {m['sharpe']:>8.2f}")
        print(f"  最大回撤: {m['max_drawdown']*100:>8.2f}%")
        print(f"  胜率:     {m['win_rate']*100:>7.1f}%")

    if len(all_results) < 4:
        print(f"\n⚠️ 只有 {len(all_results)}/4 折完成, 等全部完成再汇总")
        return

    # 汇总
    print("\n" + "=" * 70)
    print("汇总统计 (4折)")
    print("=" * 70)

    total_returns = [all_results[f]['metrics']['total_return'] for f in range(4)]
    sharpes = [all_results[f]['metrics']['sharpe'] for f in range(4)]
    max_dds = [all_results[f]['metrics']['max_drawdown'] for f in range(4)]
    win_rates = [all_results[f]['metrics']['win_rate'] for f in range(4)]

    print(f"\n累计收益: {[f'{r*100:.2f}%' for r in total_returns]}")
    print(f"  平均: {np.mean(total_returns)*100:.2f}%")
    print(f"夏普比率: {[f'{s:.2f}' for s in sharpes]}")
    print(f"  平均: {np.mean(sharpes):.2f}")
    print(f"最大回撤: {[f'{d*100:.2f}%' for d in max_dds]}")
    print(f"  最差: {min(max_dds)*100:.2f}%")
    print(f"胜率: {[f'{w*100:.1f}%' for w in win_rates]}")
    print(f"  平均: {np.mean(win_rates)*100:.1f}%")

    # 所有天数合并
    all_daily = []
    for f in range(4):
        all_daily.extend(all_results[f]['daily_returns'])
    all_daily = np.array(all_daily)
    cumret = np.cumprod(1 + all_daily) - 1
    print(f"\n全部{len(all_daily)}天合并:")
    print(f"  总累计收益: {cumret[-1]*100:.2f}%")
    print(f"  总夏普: {all_daily.mean()/all_daily.std()*np.sqrt(252):.2f}")
    cum_value = np.cumprod(1 + all_daily)
    peak = np.maximum.accumulate(cum_value)
    print(f"  最大回撤: {((cum_value-peak)/peak).min()*100:.2f}%")
    print(f"  胜率: {(all_daily>0).mean()*100:.1f}%")

    # 保存汇总
    summary = {
        'folds': all_results,
        'total_returns': total_returns,
        'sharpes': sharpes,
        'all_daily_returns': all_daily.tolist(),
    }
    save_path = f"{cfg.save_path}/backtest/summary_all_folds.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\n汇总已保存: {save_path}")


if __name__ == '__main__':
    main()
