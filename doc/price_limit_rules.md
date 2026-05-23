# A 股涨跌停规则与 Kronos 后处理

## 涨跌停规则总表

| 板块 | 正常涨跌幅 | ST/*ST 涨跌幅 | 新股首日规则 | 代码前缀 |
|------|-----------|--------------|------------|---------|
| 沪市主板 | ±10% | ±5% | 无限制（盘中10%停牌30分钟，20%停牌至14:57） | SH60xxxx |
| 深市主板 | ±10% | ±5% | 无限制（同沪市） | SZ00xxxx、SZ002xxx |
| 创业板 | ±20% | ±20% | 前5日无限制（盘中30%/60%各停牌10分钟） | SZ300xxx、SZ301xxx |
| 科创板 | ±20% | ±20% | 前5日无限制（同创业板） | SH688xxx、SH689xxx |
| 北交所 | ±30% | ±30% | 首日无限制 | BJ8xxxxx、BJ4xxxxx |

### 关键说明

1. **主板 ST 股票**：涨跌幅收窄至 ±5%，这是唯一 ST 影响涨跌幅的板块
2. **创业板/科创板/北交所 ST 股票**：涨跌幅与正常股票相同（分别为 ±20%、±20%、±30%）
3. **新股上市**：主板首日无涨跌幅限制，创业板/科创板前5日无限制，北交所首日无限制

## 代码自动识别

通过 `data_loader.py` 中的 `get_board_info(symbol)` 和 `get_limit_rate(symbol, is_st)` 函数自动识别：

```python
from data_loader import get_board_info, get_limit_rate, apply_price_limits

# 板块识别
board, rate = get_board_info("SH600519")    # ("沪市主板", 0.10)
board, rate = get_board_info("SZ300750")    # ("创业板", 0.20)
board, rate = get_board_info("SH688981")    # ("科创板", 0.20)
board, rate = get_board_info("BJ830799")    # ("北交所", 0.30)

# 涨跌停比例（含 ST 判断）
rate = get_limit_rate("SH600519")           # 0.10（主板）
rate = get_limit_rate("SH600519", is_st=True)  # 0.05（主板 ST）
rate = get_limit_rate("SZ300750", is_st=True)  # 0.20（创业板 ST 不变）

# 预测后处理
pred_df = apply_price_limits(pred_df, last_close, symbol="SZ300750")
pred_df = apply_price_limits(pred_df, last_close, symbol="SH600519", is_st=True)
```

## apply_price_limits 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pred_df` | DataFrame | 必填 | 预测结果 DataFrame |
| `last_close` | float | 必填 | 最后一日收盘价 |
| `limit_rate` | float | None | 手动指定涨跌停比例，覆盖自动识别 |
| `symbol` | str | None | 股票代码，用于自动识别涨跌停比例 |
| `is_st` | bool | False | 是否为 ST 股票 |

**优先级**：`limit_rate` 手动指定 > `symbol` 自动识别 > 默认 0.1

## 代码前缀识别规则

| 前缀 | 板块 | 涨跌幅 |
|------|------|--------|
| SH60 | 沪市主板 | ±10% |
| SH688 / SH689 | 科创板 | ±20% |
| SZ00 / SZ002 | 深市主板 | ±10% |
| SZ300 / SZ301 | 创业板 | ±20% |
| BJ4 / BJ8 | 北交所 | ±30% |

## Kronos 官方实践

Kronos 官方示例中提供了 A 股涨跌停后处理函数 `apply_price_limits`，默认使用 `limit_rate=0.1`（±10%），适用于主板股票。对于创业板、科创板、北交所股票，需手动传入正确的 `limit_rate`。

当前项目已升级为根据股票代码自动识别板块并应用正确的涨跌停比例，无需手动指定。

## 速查表（按涨跌幅分类）

| 涨跌幅 | 适用板块/股票类型 |
|--------|----------------|
| ±5% | 主板 ST / *ST 股票 |
| ±10% | 沪深主板普通股票 |
| ±20% | 创业板、科创板（含 ST 股票） |
| ±30% | 北交所（含 ST 股票） |
| 无限制 | 新股上市首日/前5日 |
