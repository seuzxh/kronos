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

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    renamed = {col: COLUMN_MAPPING[col.lower()] for col in df.columns if col.lower() in COLUMN_MAPPING}
    df = df.rename(columns=renamed)

    if "timestamps" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamps"])

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "amount" not in df.columns:
        df["amount"] = 0.0

    return df


def load_qlib(
    symbol: str,
    start_time: str,
    end_time: str,
    provider_uri: str = "/data02/home/zxh/qlib_local_data/cn_data",
    freq: str = "day",
) -> pd.DataFrame:
    try:
        from qlib import init as qlib_init
        from qlib.data import D
    except ImportError:
        raise ImportError(
            "pyqlib is not installed. Install it with: pip install pyqlib"
        )

    qlib_init(provider_uri=provider_uri, region="cn")

    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
    raw = D.features(
        instruments=[symbol.upper()],
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )

    if raw.empty:
        raise ValueError(f"No data returned for symbol={symbol}, start={start_time}, end={end_time}")

    raw.columns = [c.lstrip("$") for c in raw.columns]
    raw = raw.reset_index()
    raw.rename(columns={"datetime": "timestamps"}, inplace=True)

    if "instrument" in raw.columns:
        raw = raw.drop(columns=["instrument"])

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns after qlib load: {missing}")

    if "amount" not in raw.columns:
        raw["amount"] = 0.0

    return raw


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
