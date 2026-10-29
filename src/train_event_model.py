
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA = Path("data/processed/traffic_clean.csv")
CALENDAR = Path("data/raw/festival_calendar.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading traffic and calendar data...", flush=True)
df = pd.read_csv(DATA, parse_dates=["timestamp"])
calendar = pd.read_csv(CALENDAR, parse_dates=["date"])

df = df.dropna(subset=["timestamp", "duration", "route_code"])
df = df[df["duration"] > 0].copy()

# Aggregate events by date so a date with multiple events is handled safely.
calendar["is_festival"] = (
    calendar["event_type"].str.contains(
        "festival|dussehra", case=False, na=False
    ).astype(int)
)
calendar["is_public_holiday"] = pd.to_numeric(
    calendar["is_public_holiday"], errors="coerce"
).fillna(0).astype(int)

daily = calendar.groupby("date").agg(
    is_festival=("is_festival", "max"),
    is_public_holiday=("is_public_holiday", "max"),
    event_count=("event_name", "count"),
).reset_index()

df["date_only"] = df["timestamp"].dt.normalize()
daily["date"] = daily["date"].dt.normalize()

df = df.merge(
    daily,
    left_on="date_only",
    right_on="date",
    how="left",
)

for col in ["is_festival", "is_public_holiday", "event_count"]:
    df[col] = df[col].fillna(0).astype(int)

# Features known at prediction time
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df = df.sort_values("timestamp").reset_index(drop=True)
cutoff = int(len(df) * 0.8)
train, test = df.iloc[:cutoff], df.iloc[cutoff:]

features = [
    "route_code", "hour", "day_of_week", "month", "is_weekend",
    "is_festival", "is_public_holiday", "event_count"
]
numeric = features[1:]

preprocessor = ColumnTransformer([
    ("route", OneHotEncoder(handle_unknown="ignore"), ["route_code"]),
    ("numeric", SimpleImputer(strategy="median"), numeric),
])

model = make_pipeline(
    preprocessor,
    DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=20, random_state=42
    ),
)

print("Training event-aware model...", flush=True)
model.fit(train[features], train["duration"])
pred = model.predict(test[features])

mae = mean_absolute_error(test["duration"], pred)
rmse = mean_squared_error(test["duration"], pred) ** 0.5

print(f"Event-aware MAE: {mae:.2f} minutes")
print(f"Event-aware RMSE: {rmse:.2f} minutes")
print("Test period:", test["timestamp"].min(), "to", test["timestamp"].max())
print("Test records:", len(test))
print("Test records on calendar event dates:",
      int(test["is_public_holiday"].sum()))

joblib.dump(
    {"model": model, "features": features, "target": "duration"},
    MODEL_DIR / "trace_event_model.joblib",
)
print("Saved models/trace_event_model.joblib")
