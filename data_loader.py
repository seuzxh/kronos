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
    provider_uri: str = "/home/zxh/qlib_data",
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

    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"]
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

    if "amount" not in raw.columns or raw["amount"].isna().all():
        if "vwap" in raw.columns:
            raw["amount"] = raw["vwap"] * raw["volume"]
        else:
            raw["amount"] = 0.0

    if raw["amount"].isna().any():
        raw["amount"] = raw["amount"].fillna(0.0)

    return raw


def get_board_info(symbol: str) -> tuple:
    symbol = symbol.strip().upper()
    if symbol.startswith("SH"):
        code = symbol[2:]
        if code.startswith("688") or code.startswith("689"):
            return "科创板", 0.20
        elif code.startswith("60"):
            return "沪市主板", 0.10
        else:
            return "沪市其他", 0.10
    elif symbol.startswith("SZ"):
        code = symbol[2:]
        if code.startswith("300") or code.startswith("301"):
            return "创业板", 0.20
        elif code.startswith("00"):
            return "深市主板", 0.10
        else:
            return "深市其他", 0.10
    elif symbol.startswith("BJ"):
        return "北交所", 0.30
    else:
        code = symbol
        if code.startswith("688") or code.startswith("689"):
            return "科创板", 0.20
        elif code.startswith("300") or code.startswith("301"):
            return "创业板", 0.20
        elif code.startswith("60"):
            return "沪市主板", 0.10
        elif code.startswith("00"):
            return "深市主板", 0.10
        elif code.startswith("4") or code.startswith("8"):
            return "北交所", 0.30
        else:
            return "未知板块", 0.10


def get_limit_rate(symbol: str, is_st: bool = False) -> float:
    board, normal_rate = get_board_info(symbol)
    if is_st and board in ("沪市主板", "深市主板"):
        return 0.05
    return normal_rate


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = None, symbol: str = None, is_st: bool = False) -> pd.DataFrame:
    if limit_rate is None:
        if symbol:
            limit_rate = get_limit_rate(symbol, is_st=is_st)
        else:
            limit_rate = 0.1

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
