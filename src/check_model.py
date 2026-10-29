
from pathlib import Path
import joblib
import pandas as pd

model_path = Path("models/trace_baseline.joblib")
data_path = Path("data/processed/traffic_clean.csv")

bundle = joblib.load(model_path)
df = pd.read_csv(data_path)

print("Model file exists:", model_path.exists())
print("Dataset rows:", len(df))
print("Model features:", bundle["features"])
print("Target:", bundle["target"])
print("Routes in dataset:", df["route_code"].nunique())
print("Missing target values:", df["duration"].isna().sum())
