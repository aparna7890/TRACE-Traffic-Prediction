from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/processed/traffic_clean.csv")
CALENDAR = Path("data/raw/festival_calendar.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

print("1. Loading data and models...", flush=True)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
calendar = pd.read_csv(CALENDAR, parse_dates=["date"])

calendar["is_festival"] = (
    calendar["event_type"]
    .str.contains("festival|dussehra", case=False, na=False)
    .astype(int)
)
calendar["is_public_holiday"] = (
    pd.to_numeric(calendar["is_public_holiday"], errors="coerce")
    .fillna(0).astype(int)
)

daily = calendar.groupby("date").agg(
    is_festival=("is_festival", "max"),
    is_public_holiday=("is_public_holiday", "max"),
    event_count=("event_name", "count"),
).reset_index()

daily["date"] = daily["date"].dt.normalize()
df["date_only"] = df["timestamp"].dt.normalize()

df = df.merge(
    daily, left_on="date_only", right_on="date", how="left"
)

for col in ["is_festival", "is_public_holiday", "event_count"]:
    df[col] = df[col].fillna(0).astype(int)

df = df.dropna(subset=["timestamp", "duration", "route_code"])
df = df[df["duration"] > 0].copy()

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["day_of_month"] = df["timestamp"].dt.day
df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)

df = df.sort_values("timestamp").reset_index(drop=True)
cutoff = int(len(df) * 0.8)
train, test = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()

# Match the original training scripts' chronological row split.
baseline_features = [
    "route_code", "hour", "day_of_week", "month", "is_weekend"
]
context_features = baseline_features + [
    "day_of_month", "is_month_start", "is_month_end"
]
event_features = baseline_features + [
    "is_festival", "is_public_holiday", "event_count"
]

def load_model(filename):
    obj = joblib.load(MODEL_DIR / filename)
    return obj["model"] if isinstance(obj, dict) else obj

models = {
    "Decision tree baseline": (
        load_model("trace_baseline.joblib"), baseline_features
    ),
    "Calendar-position model": (
        load_model("trace_context_model.joblib"), context_features
    ),
    "Event-aware model": (
        load_model("trace_event_model.joblib"), event_features
    ),
}

print("2. Evaluating all models...", flush=True)

results = []
predictions = test[
    ["timestamp", "route_code", "duration", "event_count",
     "is_festival", "is_public_holiday"]
].copy()
predictions = predictions.rename(columns={"duration": "actual_minutes"})
predictions["is_event_day"] = (predictions["event_count"] > 0).astype(int)

for name, (model, features) in models.items():
    pred = model.predict(test[features])
    predictions[name] = pred

    for group_name, mask in [
        ("All test records", pd.Series(True, index=test.index)),
        ("Event dates", test["event_count"] > 0),
        ("Non-event dates", test["event_count"] == 0),
    ]:
        actual = test.loc[mask, "duration"]
        estimate = pd.Series(pred, index=test.index).loc[mask]

        if len(actual) == 0:
            continue

        results.append({
            "model": name,
            "group": group_name,
            "records": int(len(actual)),
            "MAE_minutes": round(mean_absolute_error(actual, estimate), 3),
            "RMSE_minutes": round(
                mean_squared_error(actual, estimate) ** 0.5, 3
            ),
        })

# Simple baseline: predict each route's mean duration from training data.
route_means = train.groupby("route_code")["duration"].mean()
route_pred = test["route_code"].map(route_means).fillna(train["duration"].mean())
predictions["Route-average baseline"] = route_pred

for group_name, mask in [
    ("All test records", pd.Series(True, index=test.index)),
    ("Event dates", test["event_count"] > 0),
    ("Non-event dates", test["event_count"] == 0),
]:
    actual = test.loc[mask, "duration"]
    estimate = route_pred.loc[mask]
    if len(actual):
        results.append({
            "model": "Route-average baseline",
            "group": group_name,
            "records": int(len(actual)),
            "MAE_minutes": round(mean_absolute_error(actual, estimate), 3),
            "RMSE_minutes": round(
                mean_squared_error(actual, estimate) ** 0.5, 3
            ),
        })

report = pd.DataFrame(results)
report.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

metadata = {
    "test_start": str(test["timestamp"].min()),
    "test_end": str(test["timestamp"].max()),
    "test_records": int(len(test)),
    "event_date_records": int((test["event_count"] > 0).sum()),
    "warning": (
        "Calendar coverage is incomplete unless independently verified. "
        "Event-date results alone do not prove causation."
    ),
}
(REPORT_DIR / "evaluation_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\n=== MODEL COMPARISON ===")
print(report.to_string(index=False))
print("\nSaved:")
print("reports/model_comparison.csv")
print("reports/test_predictions.csv")
print("reports/evaluation_metadata.json")