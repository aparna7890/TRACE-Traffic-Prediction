"""Audit a traffic CSV and produce coverage/quality reports. No ML training here."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ALIASES = {
    "date": ["date", "day"],
    "time": ["time", "local_time"],
    "route_code": ["route_code", "route", "route_id", "segment_id"],
    "duration": ["duration", "travel_time", "travel_time_minutes", "duration_min"],
    "distance": ["distance", "distance_km", "route_distance"],
}

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    renames = {}
    for canonical, options in ALIASES.items():
        if canonical in df.columns:
            continue
        for option in options:
            if option in df.columns:
                renames[option] = canonical
                break
    return df.rename(columns=renames)

def read_traffic(path: Path) -> pd.DataFrame:
    # utf-8-sig handles CSVs exported with a BOM; pandas sniffs common delimiters.
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")

def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if "date" not in df.columns:
        raise ValueError("Missing date column (expected date/time or timestamp/datetime).")
    date = df["date"].astype("string").str.strip()
    if "time" in df.columns:
        time = df["time"].astype("string").str.strip().replace({"nan": "", "<NA>": ""})
        combined = date + " " + time
        parsed = pd.to_datetime(combined, errors="coerce")
        # Date-only records are permitted; their timestamp is midnight and is flagged as date-grain.
        missing_time = time.isna() | time.eq("")
        if missing_time.any():
            parsed.loc[missing_time] = pd.to_datetime(date[missing_time], errors="coerce")
        return parsed
    return pd.to_datetime(date, errors="coerce")

def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(r"\s*km\s*$", "", regex=True).str.strip(),
        errors="coerce",
    )

def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value

def run_audit(traffic_path: Path, routes_path: Path | None, out_dir: Path) -> dict:
    if not traffic_path.exists():
        raise FileNotFoundError(
            f"Traffic CSV not found: {traffic_path}\n"
            "Download a candidate CSV from the upstream data/ directory and pass its path."
        )
    raw = read_traffic(traffic_path)
    df = normalize_columns(raw)
    original_rows = len(df)
    original_columns = list(df.columns)

    timestamp = parse_timestamp(df)
    df["timestamp"] = timestamp
    invalid_timestamp_rows = int(timestamp.isna().sum())
    df = df.loc[df["timestamp"].notna()].copy()

    if "route_code" not in df.columns:
        # Keep records auditable, but do not claim route-level modelling is possible.
        df["route_code"] = "UNKNOWN_ROUTE"

    for col in ("duration", "distance", "temp", "realfeel", "humidity", "aqi"):
        if col in df.columns:
            df[col] = safe_float(df[col])

    if "duration" not in df.columns:
        raise ValueError(
            "No travel-time/duration column found. Available columns: "
            + ", ".join(original_columns)
        )

    # Remove exact duplicate rows first; separately report duplicate route/timestamp keys.
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    duplicate_route_timestamps = int(
        df.duplicated(subset=["route_code", "timestamp"], keep=False).sum()
    )

    invalid_duration = int((df["duration"].isna() | (df["duration"] <= 0)).sum())
    invalid_distance = (
        int((df["distance"].isna() | (df["distance"] <= 0)).sum())
        if "distance" in df.columns else None
    )

    # Do not silently delete plausible outliers; flag for later review.
    if "distance" in df.columns:
        df["avg_speed_kmh"] = df["distance"] / (df["duration"] / 60.0)
    df["date_only"] = df["timestamp"].dt.date.astype(str)
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5)

    valid_time = df["timestamp"].dropna().sort_values()
    deltas = valid_time.diff().dropna().dt.total_seconds().div(60)
    positive_deltas = deltas[deltas > 0]

    per_route = (
        df.groupby("route_code")
          .agg(
              rows=("timestamp", "size"),
              first_timestamp=("timestamp", "min"),
              last_timestamp=("timestamp", "max"),
              unique_days=("date_only", "nunique"),
              unique_timestamps=("timestamp", "nunique"),
              median_duration_min=("duration", "median"),
          )
          .reset_index()
    )
    per_route["first_timestamp"] = per_route["first_timestamp"].astype(str)
    per_route["last_timestamp"] = per_route["last_timestamp"].astype(str)

    # Route metadata join coverage (if provided and compatible).
    route_join = None
    if routes_path and routes_path.exists():
        routes = normalize_columns(pd.read_csv(routes_path, encoding="utf-8-sig", sep=None, engine="python"))
        if "route_code" in routes.columns:
            route_ids = set(routes["route_code"].dropna().astype(str))
            traffic_ids = set(df["route_code"].dropna().astype(str))
            route_join = {
                "metadata_rows": int(len(routes)),
                "traffic_unique_routes": int(len(traffic_ids)),
                "traffic_routes_found_in_metadata": int(len(traffic_ids & route_ids)),
                "traffic_routes_missing_metadata": sorted(traffic_ids - route_ids),
                "metadata_routes_not_in_traffic": sorted(route_ids - traffic_ids),
                "metadata_columns": list(routes.columns),
            }
            display_cols = [c for c in ["route_code", "label_full", "label_short", "map_link"] if c in routes.columns]
            if display_cols:
                per_route = per_route.merge(routes[display_cols].drop_duplicates("route_code"), on="route_code", how="left")

    # Geography cannot be inferred safely from a filename. Report whether human-readable
    # route labels exist and require manual confirmation from the report.
    route_label_cols = [c for c in ["label_full", "label_short", "origin", "destination"] if c in per_route.columns]
    city_check = {
        "status": "manual_verification_required",
        "reason": "CSV filename and schema do not prove city coverage. Inspect route labels/Plus Codes and confirm routes are in Bengaluru.",
        "route_label_columns_available": route_label_cols,
    }

    report = {
        "source_file": str(traffic_path),
        "routes_file": str(routes_path) if routes_path and routes_path.exists() else None,
        "audit_status": "completed_with_warnings",
        "geography": city_check,
        "raw_row_count": int(original_rows),
        "raw_columns": original_columns,
        "rows_with_invalid_timestamp": invalid_timestamp_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "duplicate_route_timestamp_rows_after_exact_dedup": duplicate_route_timestamps,
        "rows_after_timestamp_parse_and_exact_dedup": int(len(df)),
        "date_min": str(df["timestamp"].min()) if len(df) else None,
        "date_max": str(df["timestamp"].max()) if len(df) else None,
        "span_days_inclusive": int((df["timestamp"].max().normalize() - df["timestamp"].min().normalize()).days + 1) if len(df) else 0,
        "unique_routes": int(df["route_code"].nunique()),
        "rows_per_route_median": float(per_route["rows"].median()) if len(per_route) else None,
        "unique_days_per_route_median": float(per_route["unique_days"].median()) if len(per_route) else None,
        "duration_missing_or_nonpositive_rows": invalid_duration,
        "distance_missing_or_nonpositive_rows": invalid_distance,
        "duration_missing_fraction": float(df["duration"].isna().mean()) if len(df) else None,
        "duration_summary_minutes": {
            "min": json_safe(df["duration"].min()) if len(df) else None,
            "median": json_safe(df["duration"].median()) if len(df) else None,
            "max": json_safe(df["duration"].max()) if len(df) else None,
        },
        "timestamp_gap_minutes_global": {
            "median": json_safe(positive_deltas.median()) if len(positive_deltas) else None,
            "p90": json_safe(positive_deltas.quantile(0.90)) if len(positive_deltas) else None,
            "max": json_safe(positive_deltas.max()) if len(positive_deltas) else None,
            "note": "Global gaps combine routes; use per-route gap analysis before interpreting sampling cadence.",
        },
        "column_missing_fraction": {str(k): float(v) for k, v in df.isna().mean().items()},
        "route_metadata_join": route_join,
        "target_selection": {
            "status": "deferred_until_coverage_review",
            "candidates": ["duration/travel time (minutes)", "average speed (km/h, only if distance is valid)", "congestion proxy only if independently defined and validated"],
            "recommendation": "Prefer travel time if city identity, temporal coverage, route consistency, and holdout volume are sufficient.",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Stable column ordering is not required; retain all source fields.
    df.to_csv(processed_dir / "traffic_clean.csv", index=False)
    per_route.to_csv(out_dir / "route_coverage.csv", index=False)
    with (out_dir / "data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", required=True, type=Path, help="Path to candidate traffic CSV")
    parser.add_argument("--routes", type=Path, default=None, help="Optional route metadata CSV")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Output reports directory")
    args = parser.parse_args()
    report = run_audit(args.traffic, args.routes, args.reports)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.reports / 'data_audit.json'}")
    print(f"Wrote {args.reports / 'route_coverage.csv'}")
    print("Target selection remains deferred until geography and coverage are verified.")

if __name__ == "__main__":
    main()
