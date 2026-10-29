
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/processed/traffic_clean.csv")
MODEL = Path("models/trace_baseline.joblib")

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.dropna(subset=["timestamp", "duration", "route_code"])
df = df[df["duration"] > 0].sort_values("timestamp").reset_index(drop=True)

cutoff = int(len(df) * 0.8)
train = df.iloc[:cutoff].copy()
test = df.iloc[cutoff:].copy()

bundle = joblib.load(MODEL)
model = bundle["model"]
features = bundle["features"]

for frame in (train, test):
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

# Simple baseline: training-period average duration for each route.
global_mean = train["duration"].mean()
route_means = train.groupby("route_code")["duration"].mean()
baseline_pred = test["route_code"].map(route_means).fillna(global_mean)

# Saved Decision Tree predictions.
model_pred = model.predict(test[features])

def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    print(f"{name}: MAE={mae:.2f} min | RMSE={rmse:.2f} min")

print("Test period:", test["timestamp"].min(), "to", test["timestamp"].max())
report("Route-average baseline", test["duration"], baseline_pred)
report("Decision Tree", test["duration"], model_pred)

print("\nPer-route Decision Tree MAE:")
for route, group in test.groupby("route_code"):
    idx = group.index
    actual = group["duration"]
    predicted = model.predict(group[features])
    print(f"{route}: {mean_absolute_error(actual, predicted):.2f} min")
