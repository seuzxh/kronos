"""
高贝塔股池 (883926.TI) 加载模块。

股池说明:
- 指数 883926.TI (同花顺),每日固定 100 只成分股,日频重平衡
- 相邻日重叠仅 ~16%,每天约 84 只进出(高换手)
- 来源: iFinD 接口 p03473,已缓存到 universe_snapshots.csv
- 列: date, code_ifind, code_qlib, name

Review 注意:
- code_qlib 统一小写化(sh600121 格式),与 qlib features 目录名一致
- blacklist 排除 4 只 factor 退化股票
- get_universe_union() 返回历史并集,供 preprocess 一次性拉数据
"""
import pandas as pd
from typing import Dict, List, Set


def load_universe(csv_path: str, blacklist: List[str] = None) -> Dict[pd.Timestamp, List[str]]:
    """
    加载股池快照,返回 {日期: [code_qlib, ...]} 字典。

    Args:
        csv_path: universe_snapshots.csv 路径
        blacklist: 需排除的 code_qlib 大写列表(如 ["SH600599", ...])

    Returns:
        dict: {pd.Timestamp(date): [code_qlib_lower, ...]}, 每日约100只

    Example:
        >>> u = load_universe("universe_snapshots.csv", ["SH600599"])
        >>> len(u)            # 609 个日期
        >>> u[pd.Timestamp("2026-07-10")]  # 当日100只股票代码
    """
    if blacklist is None:
        blacklist = []

    # 小写化黑名单以便匹配
    blacklist_lower = set(c.lower() for c in blacklist)

    df = pd.read_csv(csv_path)
    # 解析日期为 pd.Timestamp
    df['date'] = pd.to_datetime(df['date'])
    # code_qlib 统一小写 (CSV中是大写如 SH600004,qlib features 目录用小写 sh600004)
    df['code_qlib'] = df['code_qlib'].str.lower()

    # 排除黑名单
    mask = ~df['code_qlib'].isin(blacklist_lower)
    df = df[mask]

    # 按日期聚合
    universe: Dict[pd.Timestamp, List[str]] = {}
    for date, group in df.groupby('date'):
        universe[date] = sorted(group['code_qlib'].tolist())

    return universe


def get_universe_union(
    universe: Dict[pd.Timestamp, List[str]],
    start: str = None,
    end: str = None,
) -> Set[str]:
    """
    取股池在 [start, end] 区间内出现过的所有股票并集。

    用于 preprocess 阶段一次性拉取所有需要的股票数据,避免逐日查询。

    Args:
        universe: load_universe() 返回的字典
        start: 起始日期 (YYYY-MM-DD), None 表示从头
        end: 结束日期 (YYYY-MM-DD), None 表示到尾

    Returns:
        set: 股票代码集合 (小写), 预计 500-800 只
    """
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    union: Set[str] = set()
    for date, codes in universe.items():
        if start_ts and date < start_ts:
            continue
        if end_ts and date > end_ts:
            continue
        union.update(codes)

    return union


def get_universe_at_date(
    universe: Dict[pd.Timestamp, List[str]],
    date: str,
) -> List[str]:
    """
    取某一交易日的股池成分股(约100只)。

    Args:
        universe: load_universe() 返回的字典
        date: 日期 (YYYY-MM-DD)

    Returns:
        list: 股票代码列表 (小写), 若该日不在股池则返回空列表
    """
    ts = pd.Timestamp(date)
    # 精确匹配
    if ts in universe:
        return universe[ts]
    # 若无精确匹配,找最近的交易日
    available = sorted(universe.keys())
    for d in reversed(available):
        if d <= ts:
            return universe[d]
    return []


if __name__ == '__main__':
    # 自检: 加载股池并打印统计
    from config_highbeta import Config
    cfg = Config()

    print("=" * 60)
    print("股池加载自检")
    print("=" * 60)

    universe = load_universe(cfg.universe_csv, cfg.blacklist)

    dates = sorted(universe.keys())
    print(f"日期数: {len(dates)}")
    print(f"日期范围: {dates[0].date()} → {dates[-1].date()}")

    # 每日股票数统计
    daily_counts = [len(v) for v in universe.values()]
    print(f"每日股票数: min={min(daily_counts)} "
          f"max={max(daily_counts)} mean={sum(daily_counts)/len(daily_counts):.1f}")

    # 并集
    union = get_universe_union(universe)
    print(f"历史并集股票数: {len(union)}")

    # 抽样打印
    sample_date = dates[-1]
    sample_codes = universe[sample_date]
    print(f"\n抽样 ({sample_date.date()}): {sample_codes[:5]} ... ({len(sample_codes)}只)")

    # 换手率检查: 相邻日重叠
    if len(dates) >= 2:
        d1, d2 = dates[-2], dates[-1]
        s1, s2 = set(universe[d1]), set(universe[d2])
        overlap = len(s1 & s2)
        print(f"\n换手率检查 ({d1.date()}→{d2.date()}):")
        print(f"  重叠: {overlap}只, 新进: {len(s2-s1)}只, 退出: {len(s1-s2)}只")

    # 验证黑名单排除
    for bl in cfg.blacklist:
        bl_lower = bl.lower()
        found = any(bl_lower in v for v in universe.values())
        print(f"黑名单 {bl}: {'❌ 未排除(BUG!)' if found else '✅ 已排除'}")
