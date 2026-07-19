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

### 2.2 成分股 → data_pool `p03473` 接口(✅ 已实测)

高贝塔指数(883926.TI)历史成分股通过 iFinD `data_pool` 的 **`reportname=p03473`** 拉取。

**实测确认**(2026-07-20):
```python
# 接口规范(用户提供 + 实测验证)
# POST https://quantapi.51ifind.com/api/v1/data_pool
payload = {
    "reportname": "p03473",
    "functionpara": {"iv_date": "20260717", "iv_zsdm": "883926.TI"},
    "outputpara": "p03473_f001,p03473_f002,p03473_f003"
}
# 响应字段(实测确认含义):
#   p03473_f001 = 日期(如 "2026-07-17")
#   p03473_f002 = 成分股代码(100 只,如 "000014.SZ")
#   p03473_f003 = 成分股名称(中文,如 "沙河股份")
# dataVol = 300(100 股 × 3 字段)
# errorcode = 0
```

**实测样本**(2026-07-17 成分股前 5):
```
000014.SZ 沙河股份
000504.SZ 南华生物
000566.SZ 海南海药
000639.SZ ST西王
000676.SZ 智度股份
```

**关键观察**:100 只/天,与 DEVLOG 记录一致;代码格式是 iFinD 的 `000014.SZ`,需转 qlib 的 `SZ000014`。

---

## 3. 拉取脚本设计(`finetune/enhancement/fetch_data.sh`)

```bash
#!/bin/bash
# 拉取高贝塔指增方案所需数据
set -euo pipefail

# 使用 ifind conda 环境 + 2.qlib_ifind_hot_concept 的 IfindClient
PYTHON=/home/zxh/miniconda3/envs/ifind/bin/python
IFIND_LIB=/home/zxh/projects/2.qlib_ifind_hot_concept
DATA_DIR=/home/zxh/projects/Kronos/finetune/data/enhancement

mkdir -p "$DATA_DIR"

# === 1. 拉指数日线(单次请求,~1 秒)===
echo "[1/2] 拉取 883926.TI 指数日线..."
PYTHONPATH="$IFIND_LIB" $PYTHON - <<'PY'
from data.ifind_client import IfindClient, load_active_name, load_token, refresh_access_token
import pandas as pd

# active 账号可能 access_token 过期,构造前先 refresh
active = load_active_name()
access, refresh = load_token(active)
client = IfindClient(access=access, refresh=refresh)

df = client.fetch_history_quotation(
    codes=["883926.TI"],
    indicators=["open", "high", "low", "close", "volume", "amount"],
    start="2024-01-02",
    end="2026-07-17",
)
df.to_csv("finetune/data/enhancement/index_883926.csv", index=False)
print(f"✅ 指数日线: {len(df)} 行, {df['date'].min()} ~ {df['date'].max()}")
PY

# === 2. 拉成分股历史快照(逐日,609 个交易日 × 1 请求/天)===
echo "[2/2] 拉取 883926.TI 成分股历史快照(609 个交易日)..."
PYTHONPATH="$IFIND_LIB" $PYTHON finetune/enhancement/fetch_universe.py
```

### 3.1 `fetch_universe.py`(分日拉成分股,p03473)

```python
"""逐日拉 883926.TI 成分股(p03473),拼接成 universe_highbeta.csv"""
import sys, time
sys.path.insert(0, "/home/zxh/projects/2.qlib_ifind_hot_concept")
import pandas as pd
from data.ifind_client import IfindClient, load_active_name, load_token
from data import config

INDEX_CODE = "883926.TI"
START, END = "2024-01-02", "2026-07-17"
OUT = "finetune/data/enhancement/universe_highbeta.csv"

# 构造 client(用 active 账号)
active = load_active_name()
access, refresh = load_token(active)
client = IfindClient(access=access, refresh=refresh)

trade_dates = client.fetch_trade_dates(START, END)
print(f"交易日数: {len(trade_dates)} ({trade_dates[0]} ~ {trade_dates[-1]})")

all_rows = []
for i, date in enumerate(trade_dates):
    iv_date = date.replace("-", "")  # p03473 要 YYYYMMDD
    payload = {
        "reportname": "p03473",
        "functionpara": {"iv_date": iv_date, "iv_zsdm": INDEX_CODE},
        "outputpara": "p03473_f001,p03473_f002,p03473_f003",
    }
    try:
        raw = client._post(config.DATA_POOL_URL, payload)
        tbl = raw["tables"][0]["table"]
        n = len(tbl["p03473_f002"])
        for j in range(n):
            all_rows.append({
                "date": date,
                "code_ifind": tbl["p03473_f002"][j],
                "name": tbl["p03473_f003"][j],
            })
        if i % 50 == 0 or i == len(trade_dates) - 1:
            print(f"  进度: {i+1}/{len(trade_dates)} ({date}) +{n} 股")
    except Exception as e:
        print(f"  ⚠️ {date} 失败: {e},跳过")
    time.sleep(0.3)  # 友善限速,避免触发 429

result = pd.DataFrame(all_rows)
# iFinD 代码 → qlib 代码(000014.SZ → SZ000014)
result["code_qlib"] = result["code_ifind"].str[-2:] + result["code_ifind"].str[:6]
result = result[["date", "code_ifind", "code_qlib", "name"]]
result.to_csv(OUT, index=False)
print(f"✅ 成分股快照: {len(result)} 行, {result['date'].nunique()} 日")
```

**预期耗时**:609 日 × (0.3s sleep + ~0.5s 请求) ≈ **5 分钟**
**预期产物**:`universe_highbeta.csv` ~6 万行
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
| ~~成分股 reportname 找不到~~ | ~~中~~ | ✅ **已实测确认 p03473 可用**(2026-07-20)|
| iFinD 配额提前耗尽 | 低(用户确认配额正常)| 切账号 / 等重置 / 缩日期范围 |
| access_token 过期 | 高(secrets 里多数已过期)| 脚本已处理:构造 client 前 refresh;若 refresh_token 也过期需手动登录 iFinD 重新获取 |
| 指数代码 883926.TI 不可用 | ~~低~~→ ✅ **已实测可用** | — |
| 成分股历史快照不完整 | 中 | 用当时点实际可交易股票(避免幸存者偏差)|

---

## 8. 人工干预点

本 pipeline **需要人工在场**的场景:
1. **首次拉取**:确认 reportname、监控配额、处理异常
2. **配额耗尽**:手动切账号(编辑 `secrets/ifind_token.json`)
3. **数据异常**:如发现某日成分股突变为 0,需查 iFinD 是否故障

> 💡 因此方案文档把数据拉取放在 Stage 1 执行的第一步,而不是自动化跑。先手动跑通一次,再考虑自动化。
