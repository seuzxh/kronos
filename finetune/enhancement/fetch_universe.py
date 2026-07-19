"""拉取高贝塔指数(883926.TI)历史成分股快照。

数据源:iFinD data_pool reportname=p03473(实测确认 2026-07-20)
接口规范:
    POST https://quantapi.51ifind.com/api/v1/data_pool
    payload = {
        "reportname": "p03473",
        "functionpara": {"iv_date": "YYYYMMDD", "iv_zsdm": "883926.TI"},
        "outputpara": "p03473_f001,p03473_f002,p03473_f003",
    }
响应字段:
    p03473_f001 = 日期
    p03473_f002 = 成分股代码(100 只/天,000014.SZ 格式)
    p03473_f003 = 成分股名称(中文)

产物:finetune/data/enhancement/universe_highbeta.csv
schema:date, code_ifind, code_qlib, name(与 finetune/universe.py 兼容)

用法:
    cd /home/zxh/projects/Kronos
    PYTHONPATH=/home/zxh/projects/2.qlib_ifind_hot_concept \
    /home/zxh/miniconda3/envs/ifind/bin/python finetune/enhancement/fetch_universe.py
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, "/home/zxh/projects/2.qlib_ifind_hot_concept")

import pandas as pd
from data.ifind_client import IfindClient, load_active_name, load_token
from data import config


INDEX_CODE = "883926.TI"
DEFAULT_START = "2024-01-02"
DEFAULT_END = "2026-07-17"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "enhancement"
OUT_PATH = OUT_DIR / "universe_highbeta.csv"


def build_client() -> IfindClient:
    """构造 IfindClient,主动 refresh 一次避免 access_token 过期。

    secrets 里多数账号的 access_token 已过期(实测),refresh_token 也可能过期。
    构造前强制 refresh,失败则提示用户手动登录 iFinD。
    """
    active = load_active_name()
    access, refresh = load_token(active)
    print(f"[client] active 账号: {active}, access_token 前12: {access[:12]}...")

    # 主动 refresh 一次
    from data.ifind_client import refresh_access_token
    new_access = refresh_access_token(refresh, account=active)
    if not new_access:
        print(f"[client] ❌ refresh_token 也过期了。请:")
        print(f"         1. 登录 iFinD 客户端获取新 refresh_token")
        print(f"         2. 编辑 /home/zxh/projects/2.qlib_ifind_hot_concept/secrets/ifind_token.json")
        print(f"         3. 或把 active 切到其他可用账号(当前 active={active})")
        sys.exit(1)
    print(f"[client] ✅ token 刷新成功, 新 access 前12: {new_access[:12]}...")
    return IfindClient(access=new_access, refresh=refresh)


def fetch_one_day(client: IfindClient, date_iso: str) -> pd.DataFrame:
    """拉单日成分股。返回空 DataFrame 表示当日无数据(非交易日或指数未发布)。"""
    iv_date = date_iso.replace("-", "")
    payload = {
        "reportname": "p03473",
        "functionpara": {"iv_date": iv_date, "iv_zsdm": INDEX_CODE},
        "outputpara": "p03473_f001,p03473_f002,p03473_f003",
    }
    raw = client._post(config.DATA_POOL_URL, payload)
    tables = raw.get("tables") or []
    if not tables:
        return pd.DataFrame()
    tbl = tables[0].get("table") or {}
    codes = tbl.get("p03473_f002", [])
    names = tbl.get("p03473_f003", [])
    if len(codes) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "date": date_iso,
        "code_ifind": codes,
        "name": names,
    })


def ifind_to_qlib(code_ifind: str) -> str:
    """000014.SZ → SZ000014;600519.SH → SH600519"""
    return code_ifind[-2:] + code_ifind[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--sleep", type=float, default=0.3, help="请求间隔(秒),避免 429")
    ap.add_argument("--resume", action="store_true",
                    help="断点续拉:读取已有 CSV,只拉缺失日期")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = build_client()
    trade_dates = client.fetch_trade_dates(args.start, args.end)
    print(f"[plan] 交易日数: {len(trade_dates)} ({trade_dates[0]} ~ {trade_dates[-1]})")

    # 断点续拉
    done_dates = set()
    existing_rows = []
    if args.resume and OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH)
        done_dates = set(old["date"].unique())
        existing_rows = [old]
        print(f"[resume] 已有 {len(done_dates)} 个交易日,跳过;剩余 {len(trade_dates) - len(done_dates)} 个")

    todo = [d for d in trade_dates if d not in done_dates]
    print(f"[run] 待拉: {len(todo)} 个交易日\n")

    new_rows = []
    fail_count = 0
    for i, date in enumerate(todo):
        try:
            df = fetch_one_day(client, date)
            if len(df) == 0:
                print(f"  [{i+1}/{len(todo)}] {date}: 空(非交易日或指数未发布)")
                continue
            new_rows.append(df)
            if (i + 1) % 20 == 0 or i == len(todo) - 1:
                print(f"  [{i+1}/{len(todo)}] {date}: +{len(df)} 股(累计 {sum(len(r) for r in new_rows)} 股)")
        except Exception as e:
            fail_count += 1
            print(f"  ⚠️ [{i+1}/{len(todo)}] {date} 失败: {e}")
            if fail_count >= 5:
                print("  连续失败过多,停止。已拉数据先保存。")
                break
        time.sleep(args.sleep)

    # 合并保存
    all_parts = existing_rows + new_rows
    if not all_parts:
        print("❌ 无数据可保存")
        return
    result = pd.concat(all_parts, ignore_index=True)
    # 重新计算 code_qlib(确保新数据也转)
    result["code_qlib"] = result["code_ifind"].apply(ifind_to_qlib)
    result = result[["date", "code_ifind", "code_qlib", "name"]]
    result = result.drop_duplicates(subset=["date", "code_ifind"]).sort_values(["date", "code_ifind"])
    result.to_csv(OUT_PATH, index=False)

    print(f"\n✅ 完成: {OUT_PATH}")
    print(f"   总行数: {len(result)}")
    print(f"   交易日数: {result['date'].nunique()}")
    print(f"   日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"   历史并集股票数: {result['code_qlib'].nunique()}")
    if fail_count:
        print(f"   ⚠️ 失败 {fail_count} 天(可用 --resume 重跑补齐)")


if __name__ == "__main__":
    main()
