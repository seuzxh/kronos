import os
import shutil
import tempfile
import uuid
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, "/home/zxh/quant_projects/kronos")

from config import KronosConfig
from data_loader import apply_price_limits, load_csv
from llm_analyzer import LLMAnalyzer
from model import Kronos, KronosPredictor, KronosTokenizer

app = FastAPI(title="Kronos Inference API")

_config: Optional[KronosConfig] = None
_predictor: Optional[KronosPredictor] = None
_llm_analyzer: Optional[LLMAnalyzer] = None
_model_loaded: bool = False
_model_error: Optional[str] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _config, _predictor, _llm_analyzer, _model_loaded, _model_error

    _config = KronosConfig.load_config()

    errors = _config.validate()
    if errors:
        _model_error = "; ".join(errors)
        return

    try:
        device = _config.get_device()
        tokenizer = KronosTokenizer.from_pretrained(_config.kronos_tokenizer)
        model = Kronos.from_pretrained(_config.kronos_model)
        _predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=_config.kronos_max_context,
        )
        _model_loaded = True
    except Exception as exc:
        _model_error = str(exc)

    if _config.is_llm_configured():
        _llm_analyzer = LLMAnalyzer(
            api_base=_config.llm_api_base,
            model_id=_config.llm_model_id,
            api_key=_config.llm_api_key,
        )


@app.get("/api/health")
async def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.get("/api/model-status")
async def model_status() -> JSONResponse:
    loaded = _model_loaded
    model_name = _config.kronos_model if _config else None
    device = _config.get_device() if _config else None
    llm_available = _llm_analyzer.is_available() if _llm_analyzer else False

    payload: dict = {
        "loaded": loaded,
        "model": model_name,
        "device": device,
        "llm_available": llm_available,
    }

    if _model_error:
        payload["error"] = _model_error

    return JSONResponse(content=payload)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    lookback: Optional[int] = Form(None),
    pred_len: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    sample_count: Optional[int] = Form(None),
) -> JSONResponse:
    if not _model_loaded or _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    effective_lookback = lookback if lookback is not None else _config.kronos_lookback
    effective_pred_len = pred_len if pred_len is not None else _config.kronos_pred_len
    effective_temperature = temperature if temperature is not None else _config.kronos_temperature
    effective_top_p = top_p if top_p is not None else _config.kronos_top_p
    effective_sample_count = sample_count if sample_count is not None else _config.kronos_sample_count

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        df = load_csv(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.rmdir(tmp_dir)

    if len(df) < effective_lookback:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df)} rows, need at least {effective_lookback}",
        )

    x_df = df.iloc[-effective_lookback:][["open", "high", "low", "close", "volume", "amount"]]

    if "timestamps" in df.columns:
        x_timestamp = df.iloc[-effective_lookback:]["timestamps"]
        last_date = pd.Timestamp(df["timestamps"].iloc[-1])
    else:
        x_timestamp = pd.date_range(end=pd.Timestamp.now(), periods=effective_lookback, freq="B")
        last_date = x_timestamp[-1]

    y_timestamp = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=effective_pred_len,
    )

    pred_df = _predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=effective_pred_len,
        T=effective_temperature,
        top_p=effective_top_p,
        sample_count=effective_sample_count,
        verbose=False,
    )

    last_close = float(df["close"].iloc[-1])
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    analysis: Optional[dict] = None
    if _llm_analyzer is not None and _llm_analyzer.is_available():
        analysis = _llm_analyzer.analyze_prediction(
            pred_df, last_close, symbol=symbol or "",
        )

    prediction_records = pred_df.to_dict(orient="records")
    for record in prediction_records:
        for key, value in record.items():
            if isinstance(value, (pd.Timestamp,)):
                record[key] = value.isoformat()
            elif hasattr(value, "item"):
                record[key] = value.item()

    params: dict = {
        "lookback": effective_lookback,
        "pred_len": effective_pred_len,
        "temperature": effective_temperature,
        "top_p": effective_top_p,
        "sample_count": effective_sample_count,
        "symbol": symbol,
    }

    return JSONResponse(content={
        "success": True,
        "prediction": prediction_records,
        "analysis": analysis,
        "params": params,
    })
