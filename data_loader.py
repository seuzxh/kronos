import pandas as pd


COLUMN_MAPPING = {
    "date": "timestamps",
    "datetime": "timestamps",
    "timestamp": "timestamps",
    "time": "timestamps",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "close_price": "close",
    "volume": "volume",
    "vol": "volume",
    "amount": "amount",
    "amt": "amount",
    "turnover": "amount",
}

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    renamed = {col: COLUMN_MAPPING[col.lower()] for col in df.columns if col.lower() in COLUMN_MAPPING}
    df = df.rename(columns=renamed)

    if "timestamps" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamps"])

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0

    return df


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1) -> pd.DataFrame:
    result = pred_df.reset_index(drop=True).copy()
    cols = ["open", "high", "low", "close"]
    result[cols] = result[cols].astype("float64")

    prev_close = last_close
    for i in range(len(result)):
        limit_up = prev_close * (1 + limit_rate)
        limit_down = prev_close * (1 - limit_rate)
        for col in cols:
            result.at[i, col] = max(min(result.at[i, col], limit_up), limit_down)
        prev_close = float(result.at[i, "close"])

    return result
