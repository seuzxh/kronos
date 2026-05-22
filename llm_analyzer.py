from __future__ import annotations

import json
from typing import Optional

import pandas as pd

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LLMAnalyzer:
    def __init__(self, api_base: str, model_id: str, api_key: str) -> None:
        self._api_base = api_base
        self._model_id = model_id
        self._api_key = api_key

    def is_available(self) -> bool:
        if not LITELLM_AVAILABLE:
            return False
        return bool(self._api_key)

    def analyze_prediction(
        self,
        pred_df: pd.DataFrame,
        last_close: float,
        symbol: str = "",
    ) -> Optional[dict]:
        if not self.is_available():
            return None

        prompt = self._build_prompt(pred_df, last_close, symbol)

        try:
            response = litellm.completion(
                model=f"openai/{self._model_id}",
                api_base=self._api_base,
                api_key=self._api_key,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional financial analyst. "
                            "Analyze the given stock prediction data and respond in JSON format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except Exception:
            return None

        try:
            content = response.choices[0].message.content
            result = json.loads(content)
        except (AttributeError, IndexError, json.JSONDecodeError, TypeError):
            return None

        required_keys = {"trend", "support_resistance", "advice", "risk"}
        if not required_keys.issubset(result.keys()):
            return None

        return {
            "trend": str(result["trend"]),
            "support_resistance": str(result["support_resistance"]),
            "advice": str(result["advice"]),
            "risk": str(result["risk"]),
        }

    @staticmethod
    def _build_prompt(
        pred_df: pd.DataFrame, last_close: float, symbol: str
    ) -> str:
        summary_lines: list[str] = []
        summary_lines.append(f"Symbol: {symbol}" if symbol else "Symbol: N/A")
        summary_lines.append(f"Last close price (before prediction): {last_close:.4f}")
        summary_lines.append(f"Prediction length: {len(pred_df)} periods")
        summary_lines.append("")

        pred_start = float(pred_df["close"].iloc[0])
        pred_end = float(pred_df["close"].iloc[-1])
        pred_high = float(pred_df["high"].max())
        pred_low = float(pred_df["low"].min())
        change_pct = (pred_end - last_close) / last_close * 100 if last_close else 0.0

        summary_lines.append(f"Predicted close range: {pred_start:.4f} -> {pred_end:.4f}")
        summary_lines.append(f"Predicted high: {pred_high:.4f}")
        summary_lines.append(f"Predicted low: {pred_low:.4f}")
        summary_lines.append(f"Predicted change from last close: {change_pct:+.2f}%")
        summary_lines.append("")

        summary_lines.append("Prediction data (first 10 periods):")
        for idx, row in pred_df.head(10).iterrows():
            summary_lines.append(
                f"  {idx}: O={row['open']:.4f} H={row['high']:.4f} "
                f"L={row['low']:.4f} C={row['close']:.4f}"
            )
            if "volume" in pred_df.columns:
                summary_lines[-1] += f" V={row['volume']:.2f}"

        if len(pred_df) > 10:
            summary_lines.append(f"  ... ({len(pred_df) - 10} more periods)")

        data_section = "\n".join(summary_lines)

        return (
            f"{data_section}\n\n"
            "Based on the prediction data above, provide a structured analysis in JSON with these keys:\n"
            '- "trend": Overall trend interpretation (bullish/bearish/range-bound) with reasoning.\n'
            '- "support_resistance": Key support and resistance price levels identified from the prediction.\n'
            '- "advice": Investment advice (buy/sell/hold) with rationale.\n'
            '- "risk": Risk warnings and potential concerns.\n\n'
            "Respond ONLY with valid JSON."
        )
