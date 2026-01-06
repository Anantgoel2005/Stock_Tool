import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from stock_tool import (
    DEFAULT_HORIZON,
    DEFAULT_NEWS_LOOKBACK,
    DEFAULT_SYMBOL,
    get_live_prediction_with_reasoning,
    train_and_save_model,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock_webapp")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Dynamic Stock Prediction",
    description="Stylized web app for FinBERT-enhanced stock predictions.",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


class TrainPayload(BaseModel):
    symbol: str = Field(..., description="Stock ticker, e.g. HDFCBANK.NS")
    horizon: str = Field(default=DEFAULT_HORIZON, description="short_term or long_term")


class PredictPayload(BaseModel):
    symbol: str = Field(..., description="Stock ticker")
    horizon: str = Field(default=DEFAULT_HORIZON)
    news_query: Optional[str] = None
    days_back: int = Field(default=DEFAULT_NEWS_LOOKBACK, ge=1, le=30)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "defaults": {
                "symbol": DEFAULT_SYMBOL,
                "horizon": DEFAULT_HORIZON,
                "news_days": DEFAULT_NEWS_LOOKBACK,
            },
        },
    )


@app.post("/api/train")
async def train(payload: TrainPayload):
    symbol = payload.symbol.strip().upper()
    horizon = payload.horizon.strip().lower()

    if horizon not in {"short_term", "long_term"}:
        raise HTTPException(status_code=400, detail="Invalid investment horizon.")

    logger.info("Training requested for %s (%s)", symbol, horizon)
    success = train_and_save_model(symbol, investment_horizon=horizon)
    if not success:
        raise HTTPException(status_code=500, detail="Training failed. Check logs.")
    return {"status": "ok", "message": f"Model trained for {symbol} ({horizon})."}


@app.post("/api/predict")
async def predict(payload: PredictPayload):
    symbol = payload.symbol.strip().upper()
    horizon = payload.horizon.strip().lower()
    query = payload.news_query.strip() if payload.news_query else symbol
    days_back = payload.days_back or DEFAULT_NEWS_LOOKBACK

    if horizon not in {"short_term", "long_term"}:
        raise HTTPException(status_code=400, detail="Invalid investment horizon.")

    logger.info(
        "Prediction requested for %s (%s) | news '%s' (%s days)",
        symbol,
        horizon,
        query,
        days_back,
    )

    result = get_live_prediction_with_reasoning(
        symbol,
        news_query=query,
        investment_horizon=horizon,
        days_back_for_news=days_back,
        return_articles=True,
    )
    if isinstance(result, tuple):
        output, articles = result
    else:
        output, articles = result, []

    return JSONResponse({"status": "ok", "result": output, "articles": articles})

