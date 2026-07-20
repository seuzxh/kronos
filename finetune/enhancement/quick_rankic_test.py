"""快速验证 pred_len 对 RankIC 的影响(绕过 val pickle 长度限制)。

直接从 qlib bin 读完整个股数据,在验证段中间日期做预测 + 实际收益对照。
不依赖 val_daily.pkl(它长度不够算 predict_window 的实际收益)。

用法:
    cd /home/zxh/projects/Kronos/finetune
    PYTHONPATH=. python enhancement/quick_rankic_test.py --pred-len 1
    PYTHONPATH=. python enhancement/quick_rankic_test.py --pred-len 5
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent
for p in [str(FINETUNE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from enhancement.config_daily import Config
from enhancement.backtest_excess import load_model_and_tokenizer, predict_stock_returns
from enhancement.preprocess_daily import load_trade_days, fetch_stock_daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-len", type=int, default=1)
    ap.add_argument("--fold", default="full")
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--n-eval-dates", type=int, default=10)
    ap.add_argument("--sample-count", type=int, default=1)
    args = ap.parse_args()

    cfg = Config()
    cfg.predict_window = args.pred_len
    cfg.inference_sample_count = args.sample_count

    # 加载模型
    cfg_save = cfg.get_cv_save_dir(args.fold)
    cfg.finetuned_predictor_path = f"{cfg_save}/{cfg.predictor_save_folder_name}/checkpoints/best_model"
    print(f"模型: fold={args.fold}, pred_len={args.pred_len}, sample_count={args.sample_count}")

    # 加载 fold 的 val pickle(只用它的股票列表 + 验证段日期范围)
    fold_dir = cfg.get_cv_fold_dir(args.fold if args.fold != 'full' else 3)
    with open(f"{fold_dir}/val_daily.pkl", "rb") as f:
        val_data = pickle.load(f)

    # 抽样股票
    import random
    rng = random.Random(42)
    symbols = rng.sample(list(val_data.keys()), min(args.max_stocks, len(val_data)))
    print(f"评估股票: {len(symbols)} 只")

    # 读 CV 切分边界(从 cv_splits.json)
    import json
    with open(f"{cfg.dataset_path}/cv_splits.json") as f:
        splits = json.load(f)
    # full 用 fold3 验证段
    fold_n = 3 if args.fold == 'full' else int(args.fold)
    split = splits[fold_n]
    val_start = split["val_start_date"][:10]
    val_end = split["val_end_date"][:10]
    print(f"验证段: {val_start} ~ {val_end}")

    # 加载指数
    idx_df = pd.read_csv(cfg.index_csv)
    idx_df["date"] = pd.to_datetime(idx_df["date"]).dt.strftime("%Y-%m-%d")
    idx_df = idx_df.set_index("date")

    # 加载完整交易日历 + 个股完整数据(用于算实际收益)
    trade_days = load_trade_days(cfg.qlib_data_path_day)
    cal_day = pd.DatetimeIndex(trade_days).strftime("%Y-%m-%d")
    features_root = os.path.join(cfg.qlib_data_path_day, "features")

    # 验证段内取 n_eval_dates 个评估日(均匀采样,且留 pred_len 个未来日)
    val_days = [d for d in trade_days if val_start <= d <= val_end]
    # 留出 pred_len 个未来日,否则算不到实际收益
    usable_end_idx = val_days.index(val_end) - args.pred_len
    usable_val_days = val_days[:max(1, usable_end_idx)]
    if len(usable_val_days) > args.n_eval_dates:
        step = len(usable_val_days) // args.n_eval_dates
        eval_dates = usable_val_days[::step][:args.n_eval_dates]
    else:
        eval_dates = usable_val_days
    print(f"评估日数: {len(eval_dates)} (验证段内,留 {args.pred_len} 未来日)")

    # 加载模型
    predictor, device = load_model_and_tokenizer(cfg)

    # 逐评估日:读完整股票数据,预测 + 算实际超额
    print(f"\n开始回测...")
    all_records = []
    for di, eval_date in enumerate(eval_dates):
        day_recs = []
        for sym in symbols:
            # 读完整日线(用于算实际收益 + 提供 lookback 历史)
            stock_df = fetch_stock_daily(sym, cal_day, features_root,
                                         cfg.dataset_begin_time, cfg.dataset_end_time)
            if stock_df is None or eval_date not in stock_df.index:
                continue
            t_idx = stock_df.index.get_loc(eval_date)
            # 实际超额收益(t_idx 到 t_idx+pred_len)
            end_idx = t_idx + args.pred_len
            if end_idx >= len(stock_df):
                continue
            close_begin = stock_df["close"].iloc[t_idx]
            close_end = stock_df["close"].iloc[end_idx]
            # 过滤停牌/异常(NaN 或 0)
            if pd.isna(close_begin) or pd.isna(close_end) or close_begin <= 0:
                continue
            stock_ret = close_end / close_begin - 1
            try:
                idx_begin = idx_df.loc[eval_date, "idx_close"]
                idx_end_date = stock_df.index[end_idx]
                idx_end = idx_df.loc[idx_end_date, "idx_close"]
                if idx_begin <= 0 or pd.isna(idx_begin) or pd.isna(idx_end):
                    idx_ret = 0.0
                else:
                    idx_ret = idx_end / idx_begin - 1
            except (KeyError, TypeError):
                idx_ret = 0.0
            excess_ret = stock_ret - idx_ret

            # 预测(用 t_idx 之前的历史,不含 t_idx)
            history = stock_df.iloc[:t_idx].dropna(subset=['close', 'open', 'high', 'low'])
            if len(history) < cfg.lookback_window:
                continue
            pred_ret = predict_stock_returns(predictor, history, cfg)
            if pred_ret is None:
                continue
            day_recs.append({"date": eval_date, "symbol": sym,
                             "pred": pred_ret, "actual_excess": excess_ret})
        all_records.extend(day_recs)
        if day_recs:
            ic, _ = spearmanr([r["pred"] for r in day_recs],
                              [r["actual_excess"] for r in day_recs])
            print(f"  [{di+1}/{len(eval_dates)}] {eval_date}: {len(day_recs)}股, 当日RankIC={ic:+.4f}")

    if not all_records:
        print("\n❌ 无有效记录")
        return

    # 汇总
    df = pd.DataFrame(all_records)
    daily_ics = []
    for date, g in df.groupby("date"):
        if len(g) >= 10:
            ic, _ = spearmanr(g["pred"], g["actual_excess"])
            if not np.isnan(ic):
                daily_ics.append(ic)
    mean_ic = np.mean(daily_ics) if daily_ics else 0
    print(f"\n{'='*50}")
    print(f"pred_len={args.pred_len}, fold={args.fold}, {len(df)}条记录, {len(daily_ics)}评估日")
    print(f"平均 RankIC: {mean_ic:+.4f}")
    print(f"RankIC std:  {np.std(daily_ics):.4f}")
    print(f"正向占比:    {np.mean([r>0 for r in daily_ics]):.0%}")
    print(f"门禁(>0.03): {'✅ PASS' if mean_ic > 0.03 else '❌ FAIL'}")


if __name__ == "__main__":
    main()
