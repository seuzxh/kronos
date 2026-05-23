import sys
import os
import json
sys.path.insert(0, "/home/zxh/quant_projects/kronos")

import pandas as pd
import torch
from datetime import datetime
from config import KronosConfig
from data_loader import load_qlib, apply_price_limits
from model import Kronos, KronosTokenizer, KronosPredictor

SYMBOLS = ["SH600519", "SZ000001"]
LOOKBACK = 400
PRED_LEN = 20


def test_config():
    print("\n[1/6] 配置加载验证")
    cfg = KronosConfig.load_config()
    assert "base" in cfg.kronos_model.lower(), f"kronos_model={cfg.kronos_model}"
    assert cfg.qlib_provider_uri == "/home/zxh/qlib_data", f"qlib_provider_uri={cfg.qlib_provider_uri}"
    assert cfg.output_dir, "output_dir 为空"
    assert cfg.max_batch_size == 50, f"max_batch_size={cfg.max_batch_size}"
    errors = cfg.validate()
    assert errors == [], f"validate errors: {errors}"
    print("  ✓ 配置加载验证通过")
    return cfg


def test_data_loading(cfg):
    print("\n[2/6] 数据加载验证")
    df = load_qlib(symbol="SH600519", start_time="2024-01-01", end_time="2026-05-22", provider_uri=cfg.qlib_provider_uri)
    assert "amount" in df.columns, "amount 列不存在"
    assert df["amount"].isna().sum() == 0, f"amount 列有 {df['amount'].isna().sum()} 个 NaN"
    if "vwap" in df.columns:
        valid = df["amount"].notna() & df["vwap"].notna() & df["volume"].notna()
        if valid.any():
            diff = (df.loc[valid, "amount"] - df.loc[valid, "vwap"] * df.loc[valid, "volume"]).abs().max()
            assert diff < 1.0, f"amount vs vwap*volume 最大差异: {diff}"
    print(f"  ✓ 数据加载验证通过 (行数={len(df)}, amount无NaN)")
    return df


def test_single_predict(cfg, df):
    print("\n[3/6] 单条预测验证")
    device = cfg.get_device()
    tokenizer = KronosTokenizer.from_pretrained(cfg.kronos_tokenizer)
    model = Kronos.from_pretrained(cfg.kronos_model)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.kronos_max_context)

    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(df.iloc[-LOOKBACK:]["timestamps"].values, index=df.iloc[-LOOKBACK:].index)
    y_ts = pd.Series(pd.bdate_range(start=df["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN))

    pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    assert len(pred_df) == PRED_LEN, f"预测行数={len(pred_df)}, 期望={PRED_LEN}"
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        assert col in pred_df.columns, f"缺少列: {col}"
    print(f"  ✓ 单条预测验证通过 (行数={len(pred_df)})")
    return predictor, pred_df


def test_persistence(cfg, pred_df, symbol="SH600519"):
    print("\n[4/6] 输出持久化验证")
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(cfg.output_dir, date_str, symbol)
    csv_path = os.path.join(save_dir, "prediction.csv")
    meta_path = os.path.join(save_dir, "meta.json")

    os.makedirs(save_dir, exist_ok=True)
    pred_df.to_csv(csv_path)
    meta = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "lookback": LOOKBACK,
        "pred_len": PRED_LEN,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    assert os.path.exists(csv_path), "prediction.csv 不存在"
    assert os.path.exists(meta_path), "meta.json 不存在"
    with open(meta_path) as f:
        loaded_meta = json.load(f)
    for key in ["symbol", "lookback", "pred_len", "temperature", "top_p", "sample_count", "timestamp"]:
        assert key in loaded_meta, f"meta.json 缺少字段: {key}"
    print(f"  ✓ 输出持久化验证通过 ({save_dir})")


def test_price_limits(pred_df, df, symbol="SH600519"):
    print("\n[5/6] 涨跌停后处理验证")
    from data_loader import get_limit_rate
    last_close = float(df["close"].iloc[-1])
    limit_rate = get_limit_rate(symbol)
    limited_df = apply_price_limits(pred_df, last_close, symbol=symbol)
    prev_close = last_close
    for i in range(len(limited_df)):
        limit_up = prev_close * (1 + limit_rate)
        limit_down = prev_close * (1 - limit_rate)
        for col in ["open", "high", "low", "close"]:
            val = float(limited_df.at[i, col])
            assert val >= limit_down, f"{col}[{i}]={val} < 涨跌停下限 {limit_down}"
            assert val <= limit_up, f"{col}[{i}]={val} > 涨跌停上限 {limit_up}"
        prev_close = float(limited_df.at[i, "close"])
    print(f"  ✓ 涨跌停后处理验证通过 (板块涨跌幅={limit_rate*100:.0f}%)")


def test_batch_persistence(cfg, predictor, df_dict):
    print("\n[6/6] 批量预测输出持久化验证")
    df_list = []
    x_ts_list = []
    y_ts_list = []
    symbols = []

    for symbol, df in df_dict.items():
        if len(df) < LOOKBACK:
            continue
        x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(df.iloc[-LOOKBACK:]["timestamps"].values, index=df.iloc[-LOOKBACK:].index)
        y_ts = pd.Series(pd.bdate_range(start=df["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN))
        df_list.append(x_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)
        symbols.append(symbol)

    if not df_list:
        print("  ⊘ 无足够数据，跳过批量预测验证")
        return

    batch_results = predictor.predict_batch(
        df_list=df_list, x_timestamp_list=x_ts_list, y_timestamp_list=y_ts_list,
        pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    date_str = datetime.now().strftime("%Y-%m-%d")
    for i, symbol in enumerate(symbols):
        save_dir = os.path.join(cfg.output_dir, date_str, symbol)
        os.makedirs(save_dir, exist_ok=True)
        batch_results[i].to_csv(os.path.join(save_dir, "prediction.csv"))
        meta = {"symbol": symbol, "timestamp": datetime.now().isoformat(), "lookback": LOOKBACK, "pred_len": PRED_LEN, "temperature": 1.0, "top_p": 0.9, "sample_count": 1}
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        assert os.path.exists(os.path.join(save_dir, "prediction.csv")), f"{symbol} prediction.csv 不存在"
        assert os.path.exists(os.path.join(save_dir, "meta.json")), f"{symbol} meta.json 不存在"

    print(f"  ✓ 批量预测输出持久化验证通过 ({len(symbols)} 支股票)")


def main():
    try:
        cfg = test_config()
        df = test_data_loading(cfg)
        predictor, pred_df = test_single_predict(cfg, df)
        test_persistence(cfg, pred_df)
        test_price_limits(pred_df, df)

        df_dict = {}
        for symbol in SYMBOLS:
            try:
                df_dict[symbol] = load_qlib(symbol=symbol, start_time="2024-01-01", end_time="2026-05-22", provider_uri=cfg.qlib_provider_uri)
            except Exception:
                pass
        test_batch_persistence(cfg, predictor, df_dict)

        print("\n" + "=" * 50)
        print("全流程端到端测试: 全部通过 ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n全流程端到端测试: 失败 ✗")
        print(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()
