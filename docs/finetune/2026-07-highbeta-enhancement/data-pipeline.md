# 数据 Pipeline - iFinD 拉取股池与指数行情

> 本文档讲清楚"**数据怎么来**":高贝塔成分股历史快照 + 883926.TI 指数历史日线,如何从 iFinD 拉取。
> 依赖 `/home/zxh/projects/2.qlib_ifind_hot_concept/` 的 `IfindClient` 基础设施。

---

## TL;DR

- **要拉两类数据**:
  1. 高贝塔指数成分股历史快照(每个交易日 100 只,609 日)→ 用于划定训练股池
  2. 883926.TI 指数日线行情 → 用于超额收益基准 + 模型条件特征
- **iFinD 接口**:`IfindClient.fetch_history_quotation`(指数行情)+ `fetch_popularity_stock` 类似的 data_pool 拉成分股
- **配额约束**:iFinD 有周/月配额,需分批 + 切账号(详见第 4 节)

---

## 1. 数据需求清单

### 1.1 高贝塔成分股历史快照

| 字段 | 说明 | 示例 |
|---|---|---|
| date | 交易日 | 2024-01-02 |
| code_ifind | iFinD 代码 | 000002.SZ |
| code_qlib | qlib 代码 | SZ000002 |
| name | 股票名称 | 万科A |

**时间范围**:2024-01-02 ~ 2026-07-10(约 609 个交易日)
**预期产物**:`data/enhancement/universe_highbeta.csv`(约 609×100 = 6 万行)
**格式**:与 `/home/zxh/projects/3.qlib_ifind_beta/data/universe_snapshots.csv` 同 schema(兼容已有 `universe.py` 加载逻辑)

### 1.2 高贝塔指数(883926.TI)日线行情

| 字段 | 说明 |
|---|---|
| date | 交易日 |
| open / high / low / close | 开高低收(后复权)|
| volume / amount | 成交量/成交额 |

**时间范围**:2024-01-02 ~ 2026-07-10(与股池对齐)
**预期产物**:`data/enhancement/index_883926.csv`(约 609 行)

---

## 2. iFinD 接口映射

### 2.1 指数行情 → `fetch_history_quotation`

```python
client.fetch_history_quotation(
    codes=["883926.TI"],
    indicators=["open", "high", "low", "close", "volume", "amount"],
    start="2024-01-02",
    end="2026-07-10",
)
# 返回 DataFrame: date, code, open, high, low, close, volume, amount
```

> ✅ `fetch_history_quotation` 注释明确:"cmd_history_quotation(basic 网关 ft.10jqka)历史行情,**支持指数代码**"。已 LIVE-VERIFIED(000300.SH)。

### 2.2 成分股 → data_pool 接口

高贝塔指数成分股属"同花顺指数成分",iFinD 通过 data_pool 的 reportname 拉取。**关键**:需找到 883926.TI 成分股对应的 reportname。

**已知模式**(从 `fetch_popularity_stock` 推断):
```python
payload = {
    "reportname": "<成分股 reportname>",  # 待确认,可能是 ths_index_member 或类似
    "functionpara": {"date": "20240102", "indexcode": "883926.TI"},
    "outputpara": "jydm,jydm_mc"
}
# 返回 DataFrame: code_ifind, name
```

⚠️ **风险点**:具体 reportname 需查 iFinD 数据手册或现场试探。这是本方案最大的不确定性。

**降级方案**:如果 data_pool 拉成分股不通,可用 `fetch_board_map` + 历史快照近似:
- 用现有 `/home/zxh/projects/3.qlib_ifind_beta/data/universe_snapshots.csv`(虽只有 35 天)作种子
- 按"高贝塔"属性事后筛(计算个股 β vs 市场,β > 阈值纳入)

---

## 3. 拉取脚本设计(`finetune/enhancement/fetch_data.sh`)

```bash
#!/bin/bash
# 拉取高贝塔指增方案所需数据
set -euo pipefail

# 使用 ifind conda 环境(或 2.qlib_ifind_hot_concept 的 venv)
PYTHON=/home/zxh/miniconda3/envs/ifind/bin/python
IFIND_LIB=/home/zxh/projects/2.qlib_ifind_hot_concept
DATA_DIR=/home/zxh/projects/Kronos/finetune/data/enhancement

mkdir -p "$DATA_DIR"

# === 1. 拉指数日线 ===
echo "[1/2] 拉取 883926.TI 指数日线..."
PYTHONPATH="$IFIND_LIB" $PYTHON - <<'PY'
from data.ifind_client import IfindClient
import pandas as pd

client = IfindClient()
df = client.fetch_history_quotation(
    codes=["883926.TI"],
    indicators=["open", "high", "low", "close", "volume", "amount"],
    start="2024-01-02",
    end="2026-07-10",
)
df.to_csv("finetune/data/enhancement/index_883926.csv", index=False)
print(f"✅ 指数日线: {len(df)} 行, {df.iloc[:,0].min()} ~ {df.iloc[:,0].max()}")
PY

# === 2. 拉成分股历史快照 ===
echo "[2/2] 拉取 883926.TI 成分股历史快照(609 个交易日)..."
PYTHONPATH="$IFIND_LIB" $PYTHON finetune/enhancement/fetch_universe.py
```

### 3.1 `fetch_universe.py`(分日拉成分股)

```python
"""逐日拉 883926.TI 成分股,拼接成 universe_snapshots.csv"""
import pandas as pd
from data.ifind_client import IfindClient
from data.ifind_parsers import parse_data_pool

client = IfindClient()
trade_dates = client.fetch_trade_dates("2024-01-02", "2026-07-10")

all_rows = []
for i, date in enumerate(trade_dates):
    try:
        # ⚠️ reportname 待与 iFinD 确认,以下为伪代码
        df = client.fetch_index_constituents(  # 需在 IfindClient 新增薄封装
            index="883926.TI",
            date=date.replace("-", ""),
        )
        df["date"] = date
        all_rows.append(df)
        if i % 50 == 0:
            print(f"  进度: {i}/{len(trade_dates)} ({date})")
    except Exception as e:
        print(f"  ⚠️ {date} 失败: {e},跳过")

result = pd.concat(all_rows, ignore_index=True)
# 转 qlib 代码格式(SZ000002)
result["code_qlib"] = result["code_ifind"].str[-2:] + result["code_ifind"].str[:6]
result.to_csv("finetune/data/enhancement/universe_highbeta.csv", index=False)
print(f"✅ 成分股快照: {len(result)} 行, {result['date'].nunique()} 日")
```

---

## 4. 配额管理(iFinD 限制)

iFinD 有严格的周/月配额(从 `ifind_client.py` 错误码注释提取):

| 错误码 | 含义 | 应对 |
|---|---|---|
| -4301 | basic-data 周配额 5M 行耗尽 | 等下周重置 / 切账号 |
| -4307 | 单次响应过大 | 缩小 codes/日期范围分块 |
| -4308 | startdate~enddate 间隔须 <1 个月 | **必须按月分块** |
| -4318 | data 月配额耗尽 | 等月初重置 |

### 账号切换(手动)

账号池在 `2.qlib_ifind_hot_concept/secrets/ifind_token.json`,配额耗尽时:
```bash
# 编辑该文件,把 "active" 改为另一账号
vim /home/zxh/projects/2.qlib_ifind_hot_concept/secrets/ifind_token.json
```

### 本方案配额估算

| 数据 | 估算行数 | 配额占用 |
|---|---|---|
| 指数日线(609 日 × 6 字段)| ~3650 行 | 极小 |
| 成分股快照(609 日 × 100 股)| ~6 万行 | 中等 |
| **合计** | ~6.4 万行 | 占周配额 ~1.3% |

理论上周配额够。但 `-4308`(月内限制)要求成分股拉取**按月分块**(不能一次性 609 天)。

---

## 5. 数据验证(拉取后必做)

拉完跑 `finetune/enhancement/validate_data.py`:

```python
"""验证拉取数据的完整性"""
import pandas as pd

# 1. 指数日线
idx = pd.read_csv("finetune/data/enhancement/index_883926.csv")
assert len(idx) >= 590, f"指数行数不足: {len(idx)}"
assert idx["close"].isna().sum() / len(idx) < 0.01, "缺失率 > 1%"
# OHLC 一致性
assert (idx["high"] >= idx[["open","close","low"]].max(axis=1)).all(), "high 违规"
assert (idx["low"] <= idx[["open","close","high"]].min(axis=1)).all(), "low 违规"
print(f"✅ 指数日线: {len(idx)} 行")

# 2. 成分股
uni = pd.read_csv("finetune/data/enhancement/universe_highbeta.csv")
assert uni["date"].nunique() >= 590, f"交易日数不足: {uni['date'].nunique()}"
daily_counts = uni.groupby("date").size()
assert daily_counts.min() >= 80, f"某日成分股 < 80: {daily_counts.min()}"
print(f"✅ 成分股: {len(uni)} 行, {uni['date'].nunique()} 日")

# 3. 与个股 qlib 数据交叉验证
import qlib
qlib.init(provider_uri="/home/zxh/qlib_local_data/cn_data", region="cn")
from qlib.data import D
sample_stock = uni["code_qlib"].iloc[0]
df = D.features([sample_stock], ["$close"], "2024-01-02", "2026-07-10")
assert len(df) > 500, f"{sample_stock} 在 qlib 中数据不足"
print(f"✅ qlib 交叉验证通过")
```

---

## 6. 日常增量更新(数据 Loop)

```bash
# crontab:每个交易日 16:30 增量更新
30 16 * * 1-5 cd /home/zxh/projects/Kronos && \
    PYTHONPATH=/home/zxh/projects/2.qlib_ifind_hot_concept \
    /home/zxh/miniconda3/envs/ifind/bin/python \
    finetune/enhancement/incremental_update.py --days 1
```

`incremental_update.py` 拉最近 N 天的新数据,追加到 CSV,触发重训检查。

---

## 7. 风险与降级

| 风险 | 概率 | 降级方案 |
|---|---|---|
| 成分股 reportname 找不到 | 中 | 用现有 35 天快照 + β 阈值事后筛 |
| iFinD 配额提前耗尽 | 中 | 切账号 / 等重置 / 缩日期范围 |
| 指数代码 883926.TI 不可用 | 低 | 换其他高贝塔指数(如 884xxx.TI 系列)|
| 成分股历史快照不完整 | 中 | 用当时点实际可交易股票(避免幸存者偏差)|

---

## 8. 人工干预点

本 pipeline **需要人工在场**的场景:
1. **首次拉取**:确认 reportname、监控配额、处理异常
2. **配额耗尽**:手动切账号(编辑 `secrets/ifind_token.json`)
3. **数据异常**:如发现某日成分股突变为 0,需查 iFinD 是否故障

> 💡 因此方案文档把数据拉取放在 Stage 1 执行的第一步,而不是自动化跑。先手动跑通一次,再考虑自动化。
