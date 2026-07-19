"""拉取高贝塔指数(883926.TI)历史日线行情。

数据源:iFinD fetch_history_quotation(basic 网关,支持指数代码,实测确认 2026-07-20)
接口:IfindClient.fetch_history_quotation(codes, indicators, start, end)
返回列:symbol, date, open, high, low, close, volume, amount

产物:finetune/data/enhancement/index_883926.csv
注意:指数 OHLCV 是绝对值(非后复权),做超额收益时用收益率序列即可。

用法:
    cd /home/zxh/projects/Kronos
    PYTHONPATH=/home/zxh/projects/2.qlib_ifind_hot_concept \
    /home/zxh/miniconda3/envs/ifind/bin/python finetune/enhancement/fetch_index_daily.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/zxh/projects/2.qlib_ifind_hot_concept")

import pandas as pd
from data.ifind_client import (
    IfindClient, load_active_name, load_token, refresh_access_token,
)


INDEX_CODE = "883926.TI"
INDICATORS = ["open", "high", "low", "close", "volume", "amount"]
DEFAULT_START = "2024-01-02"
DEFAULT_END = "2026-07-17"

# iFinD 限制 startdate~enddate < 1 个月(-4308),按月分块
CHUNK_MONTHS = 1

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "enhancement"
OUT_PATH = OUT_DIR / "index_883926.csv"


def build_client() -> IfindClient:
    active = load_active_name()
    access, refresh = load_token(active)
    print(f"[client] active={active}, access 前12: {access[:12]}...")
    new_access = refresh_access_token(refresh, account=active)
    if not new_access:
        print("[client] ❌ refresh_token 过期,请手动登录 iFinD 获取新 token")
        sys.exit(1)
    print(f"[client] ✅ 刷新成功")
    return IfindClient(access=new_access, refresh=refresh)


def chunk_by_month(start: str, end: str) -> list[tuple[str, str]]:
    """把 [start, end] 切成不超过 1 个月的片段(-4308 限制)。"""
    from datetime import date
    from dateutil.relativedelta import relativedelta
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    chunks = []
    cur = s
    while cur <= e:
        nxt = cur + relativedelta(months=CHUNK_MONTHS) - relativedelta(days=1)
        chunk_end = min(nxt, e)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + relativedelta(days=1)
    return chunks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client()

    chunks = chunk_by_month(args.start, args.end)
    print(f"[plan] {args.start} ~ {args.end} 切成 {len(chunks)} 个月度片段")

    parts = []
    for i, (cs, ce) in enumerate(chunks, 1):
        try:
            df = client.fetch_history_quotation(
                codes=[INDEX_CODE], indicators=INDICATORS,
                start=cs, end=ce,
            )
            parts.append(df)
            print(f"  [{i}/{len(chunks)}] {cs} ~ {ce}: +{len(df)} 行")
        except Exception as e:
            print(f"  ⚠️ [{i}/{len(chunks)}] {cs} ~ {ce} 失败: {e}")

    if not parts:
        print("❌ 无数据")
        return

    result = pd.concat(parts, ignore_index=True)
    # 去重 + 排序(分块边界可能重叠)
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\n✅ 完成: {OUT_PATH}")
    print(f"   总行数: {len(result)}")
    print(f"   日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"   缺失天数: {result['close'].isna().sum()}")
    # OHLC 一致性快速检查
    bad_h = (result["high"] < result[["open", "close", "low"]].max(axis=1)).sum()
    bad_l = (result["low"] > result[["open", "close", "high"]].min(axis=1)).sum()
    print(f"   OHLC 违规: high={bad_h}, low={bad_l}")


if __name__ == "__main__":
    main()
