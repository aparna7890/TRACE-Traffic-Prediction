
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"

ROUTES_FILE = ROOT / "data" / "raw" / "csv-routes-bangalore.csv"
TRAFFIC_FILE = ROOT / "data" / "processed" / "traffic_clean.csv"

st.set_page_config(
    page_title="TRACE | Bengaluru Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 TRACE")
st.subheader("Traffic Reasoning using Auxiliary Contextual Events")
st.caption("Bengaluru • Historical traffic + calendar-aware prediction")

@st.cache_data
def load_route_metadata():
    routes = pd.read_csv(ROUTES_FILE)
    routes["route_code"] = routes["route_code"].astype(str)

    label_column = (
        "label_full" if "label_full" in routes.columns
        else "label_short" if "label_short" in routes.columns
        else "route_code"
    )

    routes["display_name"] = routes[label_column].fillna(
        routes["route_code"]
    ).astype(str)

    return dict(zip(routes["route_code"], routes["display_name"]))


@st.cache_data
def load_traffic_history():
    df = pd.read_csv(
        TRAFFIC_FILE,
        usecols=["timestamp", "route_code", "duration"],
        parse_dates=["timestamp"],
    )
    df = df.dropna(subset=["timestamp", "route_code", "duration"])
    df = df[df["duration"] > 0].copy()
    df["route_code"] = df["route_code"].astype(str)
    df["hour"] = df["timestamp"].dt.hour

    hourly = df.groupby(["route_code", "hour"])["duration"].agg(
        ["mean", "median", "count"]
    )
    overall = df.groupby("route_code")["duration"].agg(
        ["mean", "median", "count"]
    )
    return hourly, overall


try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    health.raise_for_status()
    routes_response = requests.get(f"{API_URL}/routes", timeout=5)
    routes_response.raise_for_status()
    route_codes = routes_response.json()["routes"]
    route_names = load_route_metadata()
    hourly_stats, overall_stats = load_traffic_history()
    st.success("TRACE API connected")
except Exception as exc:
    st.error(
        "Could not initialise TRACE. Ensure the API is running and "
        f"the dataset files exist. Details: {exc}"
    )
    st.stop()

st.divider()
st.header("Plan your journey")

col1, col2 = st.columns(2)

with col1:
    route_code = st.selectbox(
        "Select route",
        route_codes,
        format_func=lambda code: route_names.get(code, code),
    )

with col2:
    departure_date = st.date_input(
        "Departure date",
        value=datetime.now().date(),
    )

departure_time = st.time_input(
    "Departure time",
    value=datetime.strptime("08:00", "%H:%M").time(),
)

if st.button("Predict journey", type="primary", use_container_width=True):
    departure = datetime.combine(departure_date, departure_time)

    payload = {
        "route_code": route_code,
        "departure_time": departure.isoformat(),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=15
        )

        if not response.ok:
            st.error(f"Prediction failed: {response.text}")
            st.stop()

        result = response.json()
        predicted = result["predicted_travel_time_minutes"]
        context = result["calendar_context"]

        st.divider()
        st.header("Your TRACE prediction")

        m1, m2 = st.columns(2)
        m1.metric("Predicted travel time", f"{predicted:.1f} min")
        m2.metric("Departure time", departure.strftime("%d %b %Y, %I:%M %p"))

        st.write("**Route**")
        st.write(route_names.get(route_code, route_code))

        st.subheader("Why this estimate?")

        # Compare this hour's historical pattern with the route's usual average.
        key = (route_code, departure.hour)

        if key in hourly_stats.index and route_code in overall_stats.index:
            hour_data = hourly_stats.loc[key]
            route_data = overall_stats.loc[route_code]

            typical_hour = float(hour_data["mean"])
            route_average = float(route_data["mean"])
            sample_count = int(hour_data["count"])

            delta_pct = (
                (typical_hour - route_average) / route_average * 100
                if route_average > 0 else 0
            )

            a, b, c = st.columns(3)
            a.metric(
                "Historical average at this hour",
                f"{typical_hour:.1f} min",
            )
            b.metric(
                "Route's overall average",
                f"{route_average:.1f} min",
            )
            c.metric("Historical observations", f"{sample_count:,}")

            if delta_pct >= 15:
                st.warning(
                    f"Historically, this route takes about "
                    f"{delta_pct:.0f}% longer at this hour than its "
                    "overall average. This suggests a higher-traffic "
                    "period in the historical data."
                )
            elif delta_pct <= -15:
                st.success(
                    f"Historically, this route takes about "
                    f"{abs(delta_pct):.0f}% less time at this hour "
                    "than its overall average."
                )
            else:
                st.info(
                    "This hour's historical average is close to the "
                    "route's overall average."
                )

            st.caption(
                "This comparison uses historical observations for the "
                "selected route and hour. It is not a live traffic reading."
            )
        else:
            st.info(
                "There is not enough historical data for this route "
                "and hour to provide an hourly comparison."
            )

        st.subheader("Calendar context")

        c1, c2, c3 = st.columns(3)
        c1.metric("Festival flagged", "Yes" if context["is_festival"] else "No")
        c2.metric(
            "Public holiday flagged",
            "Yes" if context["is_public_holiday"] else "No",
        )
        c3.metric("Events listed", context["event_count"])

        if context["event_count"] > 0:
            st.info(
                "The calendar contains an event on this date. The model "
                "uses the calendar indicators, but this does not prove "
                "that the event caused a change in traffic."
            )
        else:
            st.caption(
                "No event is currently listed for this date. An incomplete "
                "calendar can miss events."
            )

        st.subheader("Prediction vs historical pattern")
        st.write(
            f"Model estimate: **{predicted:.1f} minutes**. "
            "The historical hourly average above provides context; "
            "it is not a second prediction from the model."
        )

        st.caption(result["note"])

    except requests.RequestException as exc:
        st.error(f"Could not contact TRACE API: {exc}")

st.divider()
st.caption(
    "TRACE is an experimental research prototype. It uses historical "
    "traffic observations and a manually maintained event calendar. "
    "It does not currently use live traffic feeds."
)
