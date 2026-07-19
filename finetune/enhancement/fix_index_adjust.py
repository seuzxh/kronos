"""修复高贝塔指数(883926.TI)的后复权口径跳变。

问题:iFinD 返回的指数 close 在某些日期出现非真实的跳变(如 2026-05-25 的 -99.96%),
原因是指数样本调整或复权基数切换,不是真实价格变动。

修复策略:
1. 计算日收益率序列
2. 识别口径切换日:|日收益| > 阈值(默认 15%,A 股指数单日涨跌幅极少超过 10%)
3. 把这些异常日的日收益设为 0(视为"无变化的口径切换")
4. 从首日 close 重建连续净值序列
5. OHLC 按相同比例缩放(保持日内 OHLC 关系)

注意:修复后 close 的绝对值无意义(只是相对基准),但日收益率序列有效。
对 Kronos 训练和超额收益计算,只需相对变化,所以修复后数据可用。

用法:
    cd /home/zxh/projects/Kronos
    python finetune/enhancement/fix_index_adjust.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

INDEX_CSV = Path(__file__).resolve().parents[1] / "data" / "enhancement" / "index_883926.csv"
THRESHOLD = 0.15  # |日收益| > 15% 视为口径跳变


def main():
    print(f"修复指数口径跳变: {INDEX_CSV}")
    df = pd.read_csv(INDEX_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  原始: {len(df)} 行, close {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")

    # 日收益
    df["ret"] = df["close"].pct_change()
    df.loc[df.index[0], "ret"] = 0.0

    # 识别异常
    abnormal_mask = df["ret"].abs() > THRESHOLD
    abnormal_dates = df.loc[abnormal_mask, ["date", "close", "ret"]]
    print(f"\n  识别到 {len(abnormal_dates)} 个口径跳变日(阈值 |ret|>{THRESHOLD*100:.0f}%):")
    print(abnormal_dates.to_string(index=False))

    # 修复:异常日 ret 置 0
    df["ret_fixed"] = df["ret"].where(~abnormal_mask, 0.0)

    # 重建净值(从首日 close 起)
    base = df["close"].iloc[0]
    scale = base * (1 + df["ret_fixed"]).cumprod() / df["close"]
    # OHLC 按比例缩放(保持日内关系)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * scale
    # volume / amount 不变(量能不受复权影响)

    # 清理临时列
    df = df.drop(columns=["ret", "ret_fixed"])

    # 备份原文件 + 覆盖
    backup = INDEX_CSV.with_suffix(".csv.raw_backup")
    if not backup.exists():
        df_orig = pd.read_csv(INDEX_CSV)  # 重新读未被修改的原始
        df_orig.to_csv(backup, index=False)
        print(f"\n  原始数据备份: {backup}")

    df.to_csv(INDEX_CSV, index=False)

    # 验证
    df2 = pd.read_csv(INDEX_CSV)
    df2["date"] = pd.to_datetime(df2["date"])
    df2["ret"] = df2["close"].pct_change()
    print(f"\n  修复后: close {df2['close'].iloc[0]:.2f} → {df2['close'].iloc[-1]:.2f}")
    print(f"  日收益 std: {df2['ret'].std()*100:.3f}%")
    print(f"  max |ret|: {df2['ret'].abs().max()*100:.3f}%")
    print(f"  异常日数(>15%): {(df2['ret'].abs() > THRESHOLD).sum()}")
    print(f"\n✅ 修复完成。新数据已覆盖原 CSV。")


if __name__ == "__main__":
    main()
