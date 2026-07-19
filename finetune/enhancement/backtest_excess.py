"""
高贝塔指增 DAILY 回测 - 超额收益 + RankIC 评估。

本模块是修正上次实验"Val Loss 低却不赚钱"的核心:
- 主指标:超额 RankIC(Spearman 排序相关,衡量横截面区分度)
- 信号:预测个股未来收益 − 实际指数未来收益 = 超额收益
- 不再只看 Val Loss,而是看模型预测的横截面排序能力

回测流程:
1. 对验证段每个交易日 t:
   - 取每只成分股过去 lookback 日(到 t-1)的真实 K 线
   - 用微调后的 Kronos 预测未来 predict 日
   - 计算预测收益 = pred_close[-1] / pred_close[0] - 1
   - 反归一化得到真实预测收益
2. 当日实际超额收益 = 真实个股收益 - 真实指数收益(对应 predict 窗口)
3. RankIC = Spearman(预测超额排序, 实际超额排序)
4. 选 top-K,累计实际超额收益,扣手续费

用法:
    cd /home/zxh/projects/Kronos/finetune
    PYTHONPATH=. python enhancement/backtest_excess.py --fold 0
    PYTHONPATH=. python enhancement/backtest_excess.py --fold full
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent
# model 包在项目根目录,finetune 代码在 finetune/
for p in [str(FINETUNE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model_and_tokenizer(cfg):
    """加载微调后的 predictor + 预训练 tokenizer"""
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 预训练 tokenizer
    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path)
    tokenizer.to(device).eval()

    # 微调后的 predictor
    model_path = cfg.finetuned_predictor_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型不存在: {model_path}\n请先训练: KRONOS_CONFIG=config_daily bash enhancement/run_daily.sh train <fold>"
        )
    model = Kronos.from_pretrained(model_path)
    model.to(device).eval()

    predictor = KronosPredictor(
        model=model, tokenizer=tokenizer, device=device,
        max_context=cfg.max_context, clip=cfg.clip,
    )
    return predictor, device


def predict_stock_returns(predictor, stock_history_df, cfg):
    """对单只股票预测未来 predict_window 日的收益。

    Args:
        predictor: KronosPredictor 实例
        stock_history_df: 历史日线 DataFrame(index=date, cols=OHLCV),至少 lookback 行
        cfg: 配置

    Returns:
        predicted_return: 预测的 predict_window 日累计收益(close_end/close_begin - 1)
                          若预测失败返回 None
    """
    import torch

    lookback = cfg.lookback_window
    if len(stock_history_df) < lookback:
        return None

    # 取最近 lookback 日
    df = stock_history_df.iloc[-lookback:].copy()

    # 构造输入(KronosPredictor.predict 需要 OHLCV DataFrame)
    # 注意:KronosPredictor 内部用的列名是 'volume'/'amount',不是 Kronos 微调约定的 'vol'/'amt'
    # 这里做列名映射,避免量能被自动填 0
    price_cols = ['open', 'high', 'low', 'close']
    for c in price_cols:
        if c not in df.columns:
            return None

    x_df = df[price_cols].copy()
    # 列名映射:vol→volume, amt→amount
    if 'vol' in df.columns:
        x_df['volume'] = df['vol']
    elif 'volume' in df.columns:
        x_df['volume'] = df['volume']
    else:
        x_df['volume'] = 0.0
    if 'amt' in df.columns:
        x_df['amount'] = df['amt']
    elif 'amount' in df.columns:
        x_df['amount'] = df['amount']
    else:
        x_df['amount'] = x_df['volume'] * x_df[price_cols].mean(axis=1)

    # 不能有 NaN(predict 会 raise)
    if x_df.isnull().values.any():
        return None
    x_df = x_df.astype(np.float32)

    # 时间戳:calc_time_stamps 用 .dt.minute 访问器,要求传入 pandas Series(不是 Index)
    timestamps = pd.Series(pd.to_datetime(df.index))

    # 未来时间戳:calc_time_stamps 用 .dt 访问器,要求 pandas Series
    # 用最后一天的 weekday 推断未来 predict_window 个交易日(跳周末)
    last_date = pd.to_datetime(df.index[-1])
    future_dates = []
    d = last_date
    while len(future_dates) < cfg.predict_window:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:  # 跳周末
            future_dates.append(d)
    future_ts = pd.Series(pd.to_datetime(future_dates))

    try:
        with torch.no_grad():
            pred_df = predictor.predict(
                x_df.reset_index(drop=True),
                x_timestamp=timestamps,
                y_timestamp=future_ts,
                pred_len=cfg.predict_window,
                T=cfg.inference_T,
                top_p=cfg.inference_top_p,
                top_k=cfg.inference_top_k,
                sample_count=cfg.inference_sample_count,
                verbose=False,
            )
        # pred_df 是预测的 K 线 DataFrame,含 open/high/low/close 列
        pred_close = pred_df["close"].values
        if len(pred_close) < 2:
            return None
        # 预测累计收益 = 末日收盘 / 首日收盘 - 1
        pred_return = float(pred_close[-1] / max(pred_close[0], 1e-6) - 1)
        return pred_return
    except Exception as e:
        # 打印首个股票的错误(调试用);生产时把 verbose 关掉
        import traceback
        if os.environ.get("BACKTEST_DEBUG"):
            print(f"    [predict error] {type(e).__name__}: {e}")
            traceback.print_exc()
        return None


def compute_actual_excess_returns(stock_df, idx_df, t_idx, predict_window):
    """计算从 t_idx 起 predict_window 日的实际超额收益。

    Args:
        stock_df: 个股完整日线 DataFrame
        idx_df: 指数完整日线 DataFrame
        t_idx: 起始日在 stock_df.index 中的位置
        predict_window: 预测窗口

    Returns:
        (stock_return, index_return, excess_return) 或 None
    """
    end_idx = t_idx + predict_window
    if end_idx >= len(stock_df):
        return None

    # 个股实际收益:用真实 close 算
    stock_close_begin = stock_df["close"].iloc[t_idx]
    stock_close_end = stock_df["close"].iloc[end_idx]
    if pd.isna(stock_close_begin) or pd.isna(stock_close_end) or stock_close_begin <= 0:
        return None
    stock_return = stock_close_end / stock_close_begin - 1

    # 指数实际收益(同期)
    date_begin = stock_df.index[t_idx]
    date_end = stock_df.index[end_idx]
    try:
        idx_begin = idx_df.loc[date_begin, "idx_close"]
        idx_end = idx_df.loc[date_end, "idx_close"]
        if pd.isna(idx_begin) or pd.isna(idx_end) or idx_begin <= 0:
            index_return = 0.0
        else:
            index_return = idx_end / idx_begin - 1
    except (KeyError, AttributeError):
        index_return = 0.0

    excess_return = stock_return - index_return
    return (stock_return, index_return, excess_return)


def backtest_fold(fold, cfg):
    """对单折做超额收益回测 + RankIC 评估"""
    import torch

    print(f"\n{'='*60}")
    print(f"回测 fold={fold}")
    print(f"{'='*60}")

    # 1. 加载验证段数据(直接用 preprocess 产物的 val_daily.pkl)
    fold_dir = cfg.get_cv_fold_dir(fold)
    val_pkl = f"{fold_dir}/val_daily.pkl"
    if not os.path.exists(val_pkl):
        # full 模型没有 val,复用 fold3 的验证段(与上次实验一致)
        if fold == 'full':
            val_pkl = f"{cfg.get_cv_fold_dir(3)}/val_daily.pkl"
            print(f"  [full 模型] 复用 fold3 验证段: {val_pkl}")
        if not os.path.exists(val_pkl):
            raise FileNotFoundError(f"验证数据不存在: {val_pkl}")

    import pickle
    with open(val_pkl, "rb") as f:
        val_data = pickle.load(f)
    print(f"  验证股票数: {len(val_data)}")

    # 2. 加载指数行情
    idx_df = pd.read_csv(cfg.index_csv)
    idx_df["date"] = pd.to_datetime(idx_df["date"]).dt.strftime("%Y-%m-%d")
    idx_df = idx_df.set_index("date")

    # 3. 加载模型
    print(f"  加载模型: {cfg.finetuned_predictor_path}")
    predictor, device = load_model_and_tokenizer(cfg)

    # 4. 逐日回测:验证段每个交易日,对所有成分股预测
    # 收集所有 (date, stock, pred_return, actual_excess)
    lookback = cfg.lookback_window
    predict_window = cfg.predict_window

    # 取验证段的日期范围(从 pickle 第一只股票的 index 推断)
    sample_stock = next(iter(val_data.values()))
    val_dates = sample_stock.index.tolist()

    # 验证段每个交易日做一次预测(为了控制时间,每隔 N 天采样)
    # predict_window=5,所以每 5 天预测一次避免重叠
    eval_dates = val_dates[lookback::predict_window]
    print(f"  验证日期数: {len(eval_dates)} (每 {predict_window} 日采样)")

    all_records = []
    for di, eval_date in enumerate(eval_dates):
        day_records = []
        for symbol, stock_df in val_data.items():
            if eval_date not in stock_df.index:
                continue
            t_idx = stock_df.index.get_loc(eval_date)
            # 实际超额收益
            actual = compute_actual_excess_returns(stock_df, idx_df, t_idx, predict_window)
            if actual is None:
                continue
            stock_ret, idx_ret, excess_ret = actual
            # 预测:用 t_idx 之前的历史(不包含 t_idx,严格防泄露)
            history = stock_df.iloc[:t_idx]
            pred_ret = predict_stock_returns(predictor, history, cfg)
            if pred_ret is None:
                continue
            day_records.append({
                "date": eval_date,
                "symbol": symbol,
                "pred_return": pred_ret,
                "actual_stock_return": stock_ret,
                "actual_index_return": idx_ret,
                "actual_excess_return": excess_ret,
            })
        all_records.extend(day_records)
        if (di + 1) % 5 == 0:
            print(f"  [{di+1}/{len(eval_dates)}] {eval_date}: {len(day_records)} 股预测完成")

    if not all_records:
        print("  ❌ 无有效回测记录")
        return None

    df = pd.DataFrame(all_records)
    print(f"\n  总记录数: {len(df)}")

    # 5. 计算指标
    # 5.1 每日 RankIC(横截面 Spearman)
    daily_rankic = []
    daily_records_groups = []
    for date, group in df.groupby("date"):
        if len(group) < 10:  # 样本太少 RankIC 无意义
            continue
        ic, _ = spearmanr(group["pred_return"], group["actual_excess_return"])
        if not np.isnan(ic):
            daily_rankic.append(ic)
        daily_records_groups.append((date, group))

    mean_rankic = np.mean(daily_rankic) if daily_rankic else 0.0
    rankic_std = np.std(daily_rankic) if daily_rankic else 0.0
    rankic_ir = mean_rankic / rankic_std if rankic_std > 0 else 0.0

    # 5.2 top-K 选股累计超额收益
    top_k = cfg.backtest_top_k
    cost_bps = cfg.backtest_cost_bps
    daily_excess_pnl = []
    for date, group in daily_records_groups:
        # 按 pred_return 降序选 top-K
        top = group.nlargest(top_k, "pred_return")
        # 平均实际超额收益
        avg_excess = top["actual_excess_return"].mean()
        # 扣手续费(每次换仓双边,简化为每 predict_window 日一次)
        cost = cost_bps / 10000.0 * 2  # 单边 bps → 双边
        net_excess = avg_excess - cost
        daily_excess_pnl.append({"date": date, "excess": net_excess})

    pnl_df = pd.DataFrame(daily_excess_pnl)
    cumulative = (1 + pnl_df["excess"]).cumprod()
    total_return = cumulative.iloc[-1] - 1
    sharpe = (pnl_df["excess"].mean() / (pnl_df["excess"].std() + 1e-9)) * np.sqrt(252 / predict_window)
    # 最大回撤
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    win_rate = (pnl_df["excess"] > 0).mean()

    # 6. 汇总输出
    summary = {
        "fold": fold,
        "n_records": len(df),
        "n_eval_days": len(daily_rankic),
        "rankic_mean": float(mean_rankic),
        "rankic_std": float(rankic_std),
        "rankic_ir": float(rankic_ir),
        "rankic_positive_ratio": float(np.mean([r > 0 for r in daily_rankic])),
        "top_k": top_k,
        "cumulative_excess_return": float(total_return),
        "excess_sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "gate_threshold": cfg.rankic_gate_threshold,
        "gate_pass": bool(mean_rankic > cfg.rankic_gate_threshold),
    }

    print(f"\n  {'指标':<25} {'值':>15}")
    print(f"  {'-'*42}")
    print(f"  {'超额 RankIC 均值':<25} {mean_rankic:>15.4f}")
    print(f"  {'超额 RankIC IR':<25} {rankic_ir:>15.4f}")
    print(f"  {'RankIC 正向占比':<25} {summary['rankic_positive_ratio']:>15.2%}")
    print(f"  {'累计超额收益':<25} {total_return:>15.2%}")
    print(f"  {'超额夏普':<25} {sharpe:>15.2f}")
    print(f"  {'最大回撤':<25} {max_drawdown:>15.2%}")
    print(f"  {'胜率':<25} {win_rate:>15.2%}")
    gate_str = "✅ PASS" if summary["gate_pass"] else "❌ FAIL"
    print(f"  {'门禁(RankIC>' + str(cfg.rankic_gate_threshold) + ')':<25} {gate_str:>15}")

    # 保存
    out_dir = os.path.join(cfg.save_path, f"fold{fold}" if fold != 'full' else 'full', "backtest")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "records.csv"), index=False)
    pnl_df.to_csv(os.path.join(out_dir, "daily_pnl.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  产物: {out_dir}")

    return summary


def main():
    import yaml  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="0", help="0/1/2/3/full")
    parser.add_argument("--all", action="store_true", help="回测全部 4 折 + full")
    args = parser.parse_args()

    from enhancement.config_daily import Config
    cfg = Config()

    if args.all:
        summaries = []
        for fold in [0, 1, 2, 3, 'full']:
            # 每折需要对应的模型路径
            cfg_save = cfg.get_cv_save_dir(fold)
            cfg.finetuned_predictor_path = f"{cfg_save}/{cfg.predictor_save_folder_name}/checkpoints/best_model"
            if not os.path.exists(cfg.finetuned_predictor_path):
                print(f"\n⚠️ fold={fold} 模型不存在,跳过: {cfg.finetuned_predictor_path}")
                continue
            s = backtest_fold(fold, cfg)
            if s:
                summaries.append(s)

        if summaries:
            # 合并 RankIC 统计
            mean_ic = np.mean([s["rankic_mean"] for s in summaries])
            print(f"\n{'='*60}")
            print(f"4 折合并: 平均 RankIC = {mean_ic:.4f}")
            pass_folds = [s["fold"] for s in summaries if s["gate_pass"]]
            print(f"门禁通过折数: {len(pass_folds)}/{len(summaries)} (fold: {pass_folds})")
            if len(pass_folds) >= 2:
                print("✅ Stage 1 门禁通过(RankIC > 0.03 持续 ≥ 2 折)→ 可进入 Stage 2 分钟 Loop")
            else:
                print("❌ Stage 1 门禁未通过 → 迭代日频 Loop(见 loop-stage1-daily.md §⑤)")
            # 保存汇总
            with open(os.path.join(cfg.save_path, "backtest_summary.json"), "w") as f:
                json.dump({"folds": summaries, "mean_rankic": float(mean_ic)}, f, indent=2, default=str)
    else:
        fold = args.fold if args.fold != "full" else "full"
        cfg_save = cfg.get_cv_save_dir(fold)
        cfg.finetuned_predictor_path = f"{cfg_save}/{cfg.predictor_save_folder_name}/checkpoints/best_model"
        backtest_fold(fold, cfg)


if __name__ == "__main__":
    main()
