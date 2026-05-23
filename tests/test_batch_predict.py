import sys
import os
sys.path.insert(0, "/home/zxh/quant_projects/kronos")

import pandas as pd
import torch
from config import KronosConfig
from data_loader import load_qlib
from model import Kronos, KronosTokenizer, KronosPredictor

SYMBOLS = ["SH600519", "SZ000001", "SH601318"]
LOOKBACK = 400
PRED_LEN = 20


def main():
    cfg = KronosConfig.load_config()
    device = cfg.get_device()

    print(f"加载模型: {cfg.kronos_model}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.kronos_tokenizer)
    model = Kronos.from_pretrained(cfg.kronos_model)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.kronos_max_context)

    df_list = []
    x_ts_list = []
    y_ts_list = []

    for symbol in SYMBOLS:
        df = load_qlib(symbol=symbol, start_time="2024-01-01", end_time="2026-05-22", provider_uri=cfg.qlib_provider_uri)
        if len(df) < LOOKBACK:
            print(f"  {symbol} 数据不足: {len(df)} < {LOOKBACK}, 跳过")
            continue
        x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(df.iloc[-LOOKBACK:]["timestamps"].values, index=df.iloc[-LOOKBACK:].index)
        y_ts = pd.Series(pd.bdate_range(start=df["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN))
        df_list.append((symbol, x_df, x_ts, y_ts))

    print("\n=== 单条预测 ===")
    single_results = {}
    for symbol, x_df, x_ts, y_ts in df_list:
        pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        single_results[symbol] = pred_df
        print(f"  {symbol}: close range [{pred_df['close'].min():.4f}, {pred_df['close'].max():.4f}]")

    print("\n=== 批量预测 ===")
    batch_x_df_list = [x_df for _, x_df, _, _ in df_list]
    batch_x_ts_list = [x_ts for _, _, x_ts, _ in df_list]
    batch_y_ts_list = [y_ts for _, _, _, y_ts in df_list]

    batch_results = predictor.predict_batch(
        df_list=batch_x_df_list,
        x_timestamp_list=batch_x_ts_list,
        y_timestamp_list=batch_y_ts_list,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    print("\n=== 批量预测结果验证 ===")
    all_pass = True
    for i, (symbol, _, _, _) in enumerate(df_list):
        batch_pred = batch_results[i]
        has_data = len(batch_pred) == PRED_LEN
        has_cols = all(col in batch_pred.columns for col in ["open", "high", "low", "close", "volume", "amount"])
        no_nan = not batch_pred[["open", "high", "low", "close"]].isna().any().any()
        passed = has_data and has_cols and no_nan
        all_pass = all_pass and passed
        status = "✓" if passed else "✗"
        print(f"  {symbol}: 行数={len(batch_pred)}, 列完整={has_cols}, 无NaN={no_nan} {status}")

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_ratio = mem_allocated / total_mem * 100
        print(f"\n=== GPU 显存 ===")
        print(f"  最大分配: {mem_allocated:.2f} GB / {total_mem:.1f} GB ({mem_ratio:.1f}%)")
        print(f"  最大保留: {mem_reserved:.2f} GB")
        mem_pass = mem_ratio < 90
        all_pass = all_pass and mem_pass
        print(f"  显存占比 < 90%: {'✓' if mem_pass else '✗'}")

    print(f"\n=== 验证结果: {'通过 ✓' if all_pass else '失败 ✗'} ===")
    return all_pass


if __name__ == "__main__":
    main()
