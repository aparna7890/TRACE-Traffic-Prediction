
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
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.dropna(subset=["timestamp", "duration", "route_code"])
df = df[df["duration"] > 0].sort_values("timestamp").reset_index(drop=True)

# Existing features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Calendar context features
df["day_of_month"] = df["timestamp"].dt.day
df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)

# IMPORTANT: This is a calendar-feature experiment, not yet a festival model.
# Actual holiday/festival labels will be added from a verified calendar next.

cutoff = int(len(df) * 0.8)
train = df.iloc[:cutoff]
test = df.iloc[cutoff:]

features = [
    "route_code", "hour", "day_of_week", "month", "is_weekend",
    "day_of_month", "is_month_start", "is_month_end"
]

preprocessor = ColumnTransformer([
    ("route", OneHotEncoder(handle_unknown="ignore"), ["route_code"]),
    ("numeric", SimpleImputer(strategy="median"), [
        "hour", "day_of_week", "month", "is_weekend",
        "day_of_month", "is_month_start", "is_month_end"
    ]),
])

model = make_pipeline(
    preprocessor,
    DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42)
)

print("Training context-feature model...", flush=True)
model.fit(train[features], train["duration"])
pred = model.predict(test[features])

mae = mean_absolute_error(test["duration"], pred)
rmse = mean_squared_error(test["duration"], pred) ** 0.5

print(f"Context-feature MAE: {mae:.2f} minutes")
print(f"Context-feature RMSE: {rmse:.2f} minutes")
print("Test period:", test["timestamp"].min(), "to", test["timestamp"].max())

joblib.dump(
    {"model": model, "features": features, "target": "duration"},
    MODEL_DIR / "trace_context_model.joblib"
)
print("Saved models/trace_context_model.joblib")
