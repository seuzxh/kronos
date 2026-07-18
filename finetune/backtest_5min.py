"""
5min 滚动预测回测脚本。

回测逻辑 (盘中动态交易, 每5min滚动预测):
  对验证段的每个交易日:
    每个交易日有48个5min节点 (9:35~15:00)
    在每个节点 t, 用过去240根5min预测未来48根, 得到预测价格路径
    根据预测路径计算每只股票的预期收益, 排序选top-K等权持有
    下一节点 t+1 按实际价格换仓

简化 (控制换手率和计算量):
  - 每个节点重新预测, 但持仓调整为 top-K (K=10)
  - 换手按 0.1% 单边手续费
  - 等权分配, 无做空

输出: 超额收益(vs等权基准), 夏普, 最大回撤, 换手率

用法:
  CUDA_VISIBLE_DEVICES=0 python backtest_5min.py --fold 0 --model-fold full
  # --fold: 验证段(0-3), --model-fold: 用哪个模型(full/0/1/2/3)
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from config_highbeta import Config
from universe import load_universe
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference


def load_model(model_fold, cfg, device):
    """加载指定 fold 的微调模型。"""
    save_dir = cfg.get_cv_save_dir(model_fold)
    model_path = f"{save_dir}/finetune_predictor/checkpoints/best_model"
    print(f"加载模型: {model_path}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path).to(device).eval()
    model = Kronos.from_pretrained(model_path).to(device).eval()
    return tokenizer, model


def prepare_stock_data(val_data, universe_dates):
    """
    准备回测数据: 按 (date, symbol) 组织, 每个股票每日的48根5min。

    Returns:
        dict[date_str -> dict[symbol -> DataFrame(48行, 5min)]]
    """
    stock_by_date = {}
    for symbol, df in val_data.items():
        # 按日分组
        for date, day_df in df.groupby(df.index.normalize()):
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in stock_by_date:
                stock_by_date[date_str] = {}
            if len(day_df) == 48:   # 只保留完整48根的日子
                stock_by_date[date_str][symbol] = day_df
    return stock_by_date


def get_stock_universe_for_date(date_str, universe_dates, stock_pool):
    """获取某日的股池成分股(约100只)。"""
    ts = pd.Timestamp(date_str)
    # 找最近的股池日期
    available = sorted(universe_dates)
    pool_codes = None
    for d in reversed(available):
        if d <= ts:
            pool_codes = set(universe_dates[d])
            break
    if pool_codes is None:
        pool_codes = set(universe_dates[available[0]])
    # 交集: 股池 ∩ 有数据的股票
    return pool_codes & stock_pool


def make_input_tensor(stock_data_today, symbols, lookback_end_idx, lookback_window, predict_window, clip_val, device):
    """
    为一批股票构造推理输入。

    Args:
        stock_data_today: dict[symbol -> DataFrame] 某日所有股票的5min数据
        symbols: 本批次要推理的股票列表
        lookback_end_idx: lookback窗口在该日5min序列中的结束位置 (0-48)
                          例如 t=10 表示用当天前10根 + 前几天的根凑240
        lookback_window: 240
        predict_window: 48

    实际上 lookback 需要240根 = 5天, 但回测是日内滚动, 我们简化为:
    用当天截至 t 时刻的所有5min根作为lookback (不足240则补0或跳过)

    更合理: 用前5天的全天5min + 当天截至t的5min = 历史窗口
    但这需要跨日数据。这里采用: 当天前 t 根作为短期lookback预测剩余。

    简化决策: 回测用"每日开盘前预测全天"模式更可行(避免跨日拼接复杂度)。
    → 见 run_daily_backtest
    """
    pass


def run_daily_prediction_backtest(
    val_data, tokenizer, model, cfg, device,
    universe_dates, val_dates_str, topk=10, fee=0.001, batch_size=4, sample_count=20,
    max_days=None
):
    """
    日级调仓回测 (基于5min预测):

    每个交易日开盘前, 对当日股池的每只股票:
      - 用前5个交易日(240根5min)作为lookback
      - 预测当日全天48根5min
      - 信号 = 预测的当日close相对前日close的涨幅
    选预测涨幅 top-K 等权买入, 当日收盘卖出(持仓1天)。

    这是最小可行的5min回测: 用5min级预测, 但日级调仓。
    每5min滚动会在这里扩展。

    Returns:
        dict: 回测结果 (日收益序列, 累计收益, 指标)
    """
    lookback = cfg.lookback_window      # 240
    pred_len = cfg.predict_window       # 48
    feature_list = cfg.feature_list     # [open,high,low,close,vol,amt]
    time_feats = cfg.time_feature_list  # [minute,hour,weekday,day,month]

    # 回测日期: 用传入的 val_dates_str (限定只回测验证段)
    dates_str = val_dates_str
    if max_days:
        dates_str = dates_str[:max_days]
    print(f"回测交易日: {len(dates_str)}天 ({dates_str[0]}~{dates_str[-1]})")

    daily_returns = []
    daily_details = []

    for di, date_str in enumerate(tqdm(dates_str, desc="回测进度")):
        date = pd.Timestamp(date_str)

        # 获取当日股池(约100只)
        # 从universe取当日成分, 再交集有数据的股票
        pool = get_stock_universe_for_date(date_str, universe_dates, set(val_data.keys()))
        pool_symbols = sorted(pool)
        if len(pool_symbols) < topk:
            continue

        # 对每只股票: 构造lookback(前240根) + 预测当日48根
        # lookback = 前几日的5min序列, 截至前一交易日收盘
        # 预测目标 = 当日48根5min
        predictions = {}   # symbol -> 预测涨幅

        # 构造 y_stamp (当日48根的时间特征, 所有股票相同)
        # 取一只股票的当日数据做模板
        sample_day_df = None
        for s in pool_symbols:
            df = val_data[s]
            day_mask = df.index.normalize() == date
            day_df = df[day_mask]
            if len(day_df) == 48:
                sample_day_df = day_df
                break
        if sample_day_df is None:
            continue
        y_ts = sample_day_df.index
        y_stamp = pd.DataFrame({
            'minute': y_ts.minute, 'hour': y_ts.hour,
            'weekday': y_ts.weekday, 'day': y_ts.day, 'month': y_ts.month
        }).values.astype(np.float32)
        y_stamp_t = torch.from_numpy(y_stamp).to(device)  # (48,5)

        # 分批推理
        valid_symbols = []
        x_batch_list = []
        for s in pool_symbols:
            df = val_data[s]
            # lookback: date之前的最后240根5min
            before_mask = df.index.normalize() < date
            hist = df[before_mask]
            if len(hist) < lookback:
                continue   # 历史不足240根, 跳过
            hist = hist.iloc[-lookback:]   # 取最近240根

            # 构造lookback特征 (6维)
            x = hist[feature_list].values.astype(np.float32)
            # lookback时间特征
            x_ts = hist.index
            x_st = pd.DataFrame({
                'minute': x_ts.minute, 'hour': x_ts.hour,
                'weekday': x_ts.weekday, 'day': x_ts.day, 'month': x_ts.month
            }).values.astype(np.float32)

            # z-score 归一化 (仅用lookback段统计量, 与训练一致)
            x_mean = x.mean(axis=0)
            x_std = x.std(axis=0)
            # 保存真实末根close (反归一化用)
            last_close_real = x[-1, 3]
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -cfg.clip, cfg.clip)

            x_batch_list.append((x_norm, x_st, x_mean[3], x_std[3], last_close_real))
            valid_symbols.append(s)

        if len(valid_symbols) < topk:
            continue

        # 分批推理 (batch_size只一批, 避免显存问题)
        for bi in range(0, len(valid_symbols), batch_size):
            batch_syms = valid_symbols[bi:bi+batch_size]
            batch_x = [x_batch_list[bi+j][0] for j in range(len(batch_syms))]
            batch_xs = [x_batch_list[bi+j][1] for j in range(len(batch_syms))]
            batch_close_std = [x_batch_list[bi+j][3] for j in range(len(batch_syms))]
            batch_last_close = [x_batch_list[bi+j][4] for j in range(len(batch_syms))]

            x_arr = np.stack(batch_x)   # (B, 240, 6)
            xs_arr = np.stack(batch_xs)  # (B, 240, 5)
            x_t = torch.from_numpy(x_arr).to(device)
            xs_t = torch.from_numpy(xs_arr).to(device)
            y_stamp_batch = y_stamp_t.unsqueeze(0).repeat(len(batch_syms), 1, 1)

            with torch.no_grad():
                preds = auto_regressive_inference(
                    tokenizer, model, x_t, xs_t, y_stamp_batch,
                    max_context=cfg.max_context, pred_len=pred_len, clip=cfg.clip,
                    T=cfg.inference_T, top_k=cfg.inference_top_k,
                    top_p=cfg.inference_top_p, sample_count=sample_count
                )
            # preds: (B, 48, 6), close在index 3
            # 信号修复: 反归一化到真实价格, 算预测涨幅
            # pred_close_real = pred_norm * std + mean
            # 涨幅 = pred_close_real / last_close_real - 1
            for j in range(len(batch_syms)):
                pred_close_norm = preds[j, -1, 3]   # 预测末根close(归一化)
                std = batch_close_std[j]
                pred_close_real = pred_close_norm * std + x_batch_list[bi+j][2]  # mean[3]
                last_real = batch_last_close[j]
                predicted_return = (pred_close_real - last_real) / (last_real + 1e-8)
                predictions[batch_syms[j]] = predicted_return

        if len(predictions) < topk:
            continue

        # 选top-K (预测涨幅最大的K只)
        ranked = sorted(predictions.items(), key=lambda x: -x[1])
        selected = [s for s, _ in ranked[:topk]]

        # 计算当日实际收益: 等权买入top-K, 用当日实际5min收盘
        # 买入价 = 当日开盘(第一根open), 卖出价 = 当日收盘(最后一根close)
        day_returns = []
        for s in selected:
            df = val_data[s]
            day_df = df[df.index.normalize() == date]
            if len(day_df) < 48:
                continue
            open_price = day_df['open'].iloc[0]
            close_price = day_df['close'].iloc[-1]
            ret = (close_price - open_price) / open_price
            day_returns.append(ret)

        if len(day_returns) == 0:
            continue
        portfolio_ret = np.mean(day_returns)

        # 扣手续费: 换仓 = 全部买卖一次
        portfolio_ret -= 2 * fee   # 买入+卖出

        daily_returns.append(portfolio_ret)
        daily_details.append({
            'date': date_str, 'return': portfolio_ret,
            'n_selected': len(day_returns),
            'top1': selected[0] if selected else '',
        })

    return daily_returns, daily_details


def compute_metrics(daily_returns, benchmark_returns=None):
    """计算回测指标。"""
    returns = np.array(daily_returns)
    n = len(returns)

    # 累计收益
    cumret = np.cumprod(1 + returns) - 1
    total_return = cumret[-1] if n > 0 else 0

    # 年化 (假设252交易日)
    if n > 0 and total_return > -1:
        ann_return = (1 + total_return) ** (252 / n) - 1
    else:
        ann_return = 0

    # 夏普 (日收益均值/标准差 * sqrt(252))
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 最大回撤
    cum_value = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_value)
    drawdown = (cum_value - peak) / peak
    max_dd = drawdown.min()

    # 超额收益 (vs benchmark)
    excess = None
    if benchmark_returns is not None:
        br = np.array(benchmark_returns[:n])
        excess_returns = returns - br
        excess_cum = np.cumprod(1 + excess_returns) - 1
        excess = excess_cum[-1] if n > 0 else 0
        if returns.std() - br.std() > 0:
            excess_sharpe = (returns - br).mean() / (returns - br).std() * np.sqrt(252)
        else:
            excess_sharpe = 0

    result = {
        'total_return': total_return,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_days': n,
        'avg_daily_return': returns.mean(),
        'win_rate': (returns > 0).mean(),
    }
    if excess is not None:
        result['excess_return'] = excess
        result['excess_sharpe'] = excess_sharpe
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='验证段 0-3')
    parser.add_argument('--model-fold', type=str, default='full',
                        help='用哪个模型: full/0/1/2/3')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--fee', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sample-count', type=int, default=20)
    parser.add_argument('--max-days', type=int, default=None, help='限制回测天数(调试用)')
    args = parser.parse_args()

    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载数据: train + val 合并 (lookback=240根=5天, 需要val段之前的train数据)
    print(f"\n[1] 加载 fold{args.fold} 数据 (train+val合并, 供lookback)...")
    fold_dir = cfg.get_cv_fold_dir(args.fold)
    with open(f"{fold_dir}/train_5min.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(f"{fold_dir}/val_5min.pkl", 'rb') as f:
        val_data = pickle.load(f)
    # 合并: train 和 val 的同一只股票拼接成完整序列
    all_data = {}
    for sym in set(list(train_data.keys()) + list(val_data.keys())):
        parts = []
        if sym in train_data:
            parts.append(train_data[sym])
        if sym in val_data:
            parts.append(val_data[sym])
        if parts:
            all_data[sym] = pd.concat(parts).sort_index()
    val_data = all_data   # 用合并后的完整数据
    print(f"  合并后股票数: {len(val_data)}")
    # 释放train_data引用(已被合并)
    del train_data

    # 2. 加载股池
    print("\n[2] 加载股池...")
    universe = load_universe(cfg.universe_csv, cfg.blacklist)
    print(f"  股池日期数: {len(universe)}")

    # 3. 加载模型
    print(f"\n[3] 加载 model_fold={args.model_fold} 模型...")
    tokenizer, model = load_model(args.model_fold, cfg, device)

    # 3.5 计算验证段日期范围 (复用 preprocess 的 CV 切分逻辑)
    from preprocess_5min import compute_cv_splits
    cal_day = pd.read_csv(
        os.path.join(cfg.qlib_data_path_day, "calendars", "day.txt"),
        header=None, names=['date']
    )
    cal_day['date'] = pd.to_datetime(cal_day['date'])
    trade_days = cal_day['date'].tolist()
    mask = [(d >= pd.Timestamp(cfg.dataset_begin_time)) and
            (d <= pd.Timestamp(cfg.dataset_end_time)) for d in trade_days]
    trade_days = [d for d, m in zip(trade_days, mask) if m]
    splits = compute_cv_splits(trade_days, cfg.cv_folds, cfg.cv_val_days)
    fold_idx = int(args.fold)
    val_start = splits[fold_idx]['val_start_date']
    val_end = splits[fold_idx]['val_end_date']
    val_dates_str = [d.strftime('%Y-%m-%d') for d in trade_days
                     if val_start <= d <= val_end]
    print(f"\n[3.5] 验证段日期: {val_dates_str[0]}~{val_dates_str[-1]} ({len(val_dates_str)}天)")

    # 4. 回测
    print(f"\n[4] 开始5min预测回测 (top{args.topk}, fee={args.fee})...")
    t0 = time.time()
    daily_returns, daily_details = run_daily_prediction_backtest(
        val_data, tokenizer, model, cfg, device,
        universe, val_dates_str, topk=args.topk, fee=args.fee,
        batch_size=args.batch_size, sample_count=args.sample_count,
        max_days=args.max_days
    )
    print(f"  回测耗时: {time.time()-t0:.0f}s")

    # 5. 计算指标
    print(f"\n[5] 回测结果:")
    metrics = compute_metrics(daily_returns)
    print(f"  回测天数: {metrics['n_days']}")
    print(f"  累计收益: {metrics['total_return']*100:.2f}%")
    print(f"  年化收益: {metrics['ann_return']*100:.2f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"  日均收益: {metrics['avg_daily_return']*100:.3f}%")
    print(f"  胜率:     {metrics['win_rate']*100:.1f}%")

    # 6. 保存结果
    save_dir = f"{cfg.save_path}/backtest"
    os.makedirs(save_dir, exist_ok=True)
    result = {
        'config': {'fold': args.fold, 'model_fold': args.model_fold,
                   'topk': args.topk, 'fee': args.fee},
        'metrics': metrics,
        'daily_returns': daily_returns,
        'daily_details': daily_details,
    }
    save_path = f"{save_dir}/backtest_fold{args.fold}_model{args.model_fold}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n  结果已保存: {save_path}")

    # 打印每日明细
    print(f"\n每日明细:")
    print(f"{'日期':<14} {'收益%':>8} {'选股数':>6} {'top1':<10}")
    for d in daily_details:
        print(f"{d['date']:<14} {d['return']*100:>8.2f} {d['n_selected']:>6} {d['top1']:<10}")


if __name__ == '__main__':
    main()
