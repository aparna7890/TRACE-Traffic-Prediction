
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

DATA = Path("data/processed/traffic_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.dropna(subset=["timestamp", "duration", "route_code"])
df = df[df["duration"] > 0].copy()

# Time features available at prediction time
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Chronological split: the latest 20% of observations are held out.
df = df.sort_values("timestamp").reset_index(drop=True)
cutoff = int(len(df) * 0.8)
train = df.iloc[:cutoff]
test = df.iloc[cutoff:]

features = ["route_code", "hour", "day_of_week", "month", "is_weekend"]
target = "duration"

preprocessor = ColumnTransformer(
    transformers=[
        ("route", OneHotEncoder(handle_unknown="ignore"), ["route_code"]),
        ("numeric", SimpleImputer(strategy="median"),
         ["hour", "day_of_week", "month", "is_weekend"]),
    ]
)

model = make_pipeline(
    preprocessor,
    RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(train[features], train[target])
predictions = model.predict(test[features])

mae = mean_absolute_error(test[target], predictions)
rmse = mean_squared_error(test[target], predictions) ** 0.5

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test period:", test["timestamp"].min(), "to", test["timestamp"].max())
print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")

joblib.dump(
    {"model": model, "features": features, "target": target},
    MODEL_DIR / "trace_baseline.joblib",
)
print("Saved model to models/trace_baseline.joblib")
