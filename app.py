
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trace_event_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "traffic_clean.csv"
CALENDAR_PATH = ROOT / "data" / "raw" / "festival_calendar.csv"

# Load the trained model and its expected feature list.
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

# Load valid Bengaluru route codes.
traffic = pd.read_csv(DATA_PATH, usecols=["route_code"])
VALID_ROUTES = sorted(traffic["route_code"].dropna().unique().tolist())

# Build a daily event lookup.
calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["date"])
calendar["is_festival"] = (
    calendar["event_type"]
    .str.contains("festival|dussehra", case=False, na=False)
    .astype(int)
)
calendar["is_public_holiday"] = (
    pd.to_numeric(calendar["is_public_holiday"], errors="coerce")
    .fillna(0).astype(int)
)

daily = calendar.groupby(calendar["date"].dt.normalize()).agg(
    is_festival=("is_festival", "max"),
    is_public_holiday=("is_public_holiday", "max"),
    event_count=("event_name", "count"),
)

app = FastAPI(
    title="TRACE API",
    description="Traffic travel-time prediction for Bengaluru",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    route_code: str = Field(
        description="Route code from the TRACE dataset"
    )
    departure_time: datetime = Field(
        description="Departure date and time, e.g. 2026-10-30T08:00:00"
    )


@app.get("/")
def home():
    return {
        "project": "TRACE",
        "description": "Traffic Reasoning using Auxiliary Contextual Events",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "trace_event_model",
        "available_routes": len(VALID_ROUTES),
    }


@app.get("/routes")
def routes():
    return {"count": len(VALID_ROUTES), "routes": VALID_ROUTES}


@app.post("/predict")
def predict(request: PredictionRequest):
    route = request.route_code.strip()
    departure = request.departure_time

    if route not in VALID_ROUTES:
        raise HTTPException(
            status_code=400,
            detail="Unknown route_code. Check GET /routes for valid codes.",
        )

    day = departure.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    event = daily.loc[day] if day in daily.index else None

    is_festival = int(event["is_festival"]) if event is not None else 0
    is_holiday = (
        int(event["is_public_holiday"]) if event is not None else 0
    )
    event_count = int(event["event_count"]) if event is not None else 0

    values = {
        "route_code": route,
        "hour": departure.hour,
        "day_of_week": departure.weekday(),
        "month": departure.month,
        "is_weekend": int(departure.weekday() >= 5),
        "is_festival": is_festival,
        "is_public_holiday": is_holiday,
        "event_count": event_count,
    }

    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    predicted = float(model.predict(row)[0])

    return {
        "project": "TRACE",
        "route_code": route,
        "departure_time": departure.isoformat(),
        "predicted_travel_time_minutes": round(max(0, predicted), 2),
        "calendar_context": {
            "is_festival": is_festival,
            "is_public_holiday": is_holiday,
            "event_count": event_count,
        },
        "model": "event-aware decision tree",
        "note": (
            "Experimental estimate based on historical data and the "
            "provided calendar. It is not a live traffic measurement."
        ),
    }
