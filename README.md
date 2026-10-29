# TRACE — Bengaluru data foundation (Milestone 1)

This starter project ingests and audits historical traffic CSVs before choosing a prediction target.
It deliberately does **not** train a model or fabricate traffic observations.

## What is verified vs pending

The upstream `thecont1/traffic-monitor-lizard` README documents timestamped route travel-time observations and weather fields, and describes its dataset licence as CC BY 4.0. Its README uses city-specific CSV filenames, but that alone does not prove that the currently published files are Bengaluru data. Confirm route geography and data coverage from the downloaded files.

Upstream: https://github.com/thecont1/traffic-monitor-lizard

## 1. Environment (Windows PowerShell)

```powershell
cd TRACE_data_pipeline
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the interpreter directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Put source files in place

Download the relevant traffic CSV and its route metadata CSV from the upstream repository's `data/` directory, then put them in:

- `data/raw/traffic.csv`
- `data/raw/routes.csv` (if available)

Do not rename a file to `traffic.csv` until you have confirmed what it contains. If the upstream files are not Bengaluru data, stop and choose a suitable source rather than treating another city as Bengaluru.

## 3. Run the audit

```powershell
python -m src.audit_data --traffic data/raw/traffic.csv --routes data/raw/routes.csv
```

If no route file exists:
```powershell
python -m src.audit_data --traffic data/raw/traffic.csv
```

The audit writes:
- `reports/data_audit.json` — machine-readable coverage and quality results
- `reports/route_coverage.csv` — row counts and date spans by route, if route IDs exist
- `data/processed/traffic_clean.csv` — cleaned, normalized records

## 4. What the audit checks

- Required columns and parsable dates/times
- Date range, row count, route count, readings per route/day
- Duplicate route-timestamp observations
- Missingness and invalid/non-positive travel times or distances
- Sampling intervals and gaps
- Route metadata join coverage, if a routes CSV is provided
- Whether the data can support hourly or daily prediction

## Prediction target is intentionally undecided

Only after the audit will we choose between:
- travel time (minutes),
- average speed (km/h, derivable from distance and duration),
- or a congestion proxy.

This source documents travel-time estimates; it does not by itself provide ground-truth traffic volume or a validated congestion label. We will not invent a congestion score. A route-level historical model also needs enough time coverage per route and a chronological holdout. Event-aware evaluation additionally needs enough independently observed festival/election/event dates.
