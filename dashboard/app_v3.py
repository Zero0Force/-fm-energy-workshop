"""FM Energy Workshop - historical electricity dashboard, v3 (replay + prediction).

Based on dashboard/app_v2.py (left unchanged). Differences from v2:
  - The Prediction area has a "Predict next hour" button.
  - It uses the SAVED model (models/gbr_model.joblib) and feature order
    (models/feature_order.json). The model is NOT retrained.
  - For each meter it builds the model inputs from that meter's own revealed
    readings only, and predicts the next hour after that meter's latest reading.
  - If required history is missing, a clear message is shown for that meter.

Everything else (January start, one new recorded hour per real minute, Restart
simulation button, stop at the last record) works as in v2.

Run from the project root (fm-energy-workshop) with the `vibe` venv:
    c:\\Users\\user\\workshop\\vibe\\Scripts\\python.exe -m streamlit run dashboard/app_v3.py --server.headless true
"""
from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

# ---------------------------------------------------------------------------
# Page config - must be the first Streamlit command.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FM Energy Dashboard (v3 - prediction)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & Neon connection setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # fm-energy-workshop
ENV_PATH = PROJECT_ROOT / ".env"
MODEL_PATH = PROJECT_ROOT / "models" / "gbr_model.joblib"
FEATURE_ORDER_PATH = PROJECT_ROOT / "models" / "feature_order.json"
TABLE = "meter_readings_source"
MAX_RETRIES = 4

INTERVAL_SEC = 60          # fetch one new hour every real minute
JAN_START = "2016-01-01 00:00:00"
JAN_END = "2016-01-31 23:00:00"

# Lag features: name -> hours before the target hour.
LAG_HOURS = {"lag_1": 1, "lag_2": 2, "lag_3": 3, "lag_24": 24, "lag_168": 168}

load_dotenv(ENV_PATH)
RAW_DB_URL = os.environ.get("DATABASE_URL")
if not RAW_DB_URL:
    st.error(f"DATABASE_URL not found in {ENV_PATH}. Add it to your .env file.")
    st.stop()


def strip_channel_binding(url: str) -> str:
    """Remove channel_binding=require and tidy any leftover separators."""
    for frag in ("&channel_binding=require", "channel_binding=require&",
                 "channel_binding=require"):
        url = url.replace(frag, "")
    url = url.replace("?&", "?").replace("&&", "&")
    if url.endswith("&"):
        url = url[:-1]
    if url.endswith("?"):
        url = url[:-1]
    return url


DB_URL = strip_channel_binding(RAW_DB_URL)
NEON_HOST = urlparse(DB_URL).hostname
SQL_ENDPOINT = f"https://{NEON_HOST}/sql"

_SESSION = requests.Session()
_HEADERS = {
    "Neon-Connection-String": DB_URL,
    "Content-Type": "application/json",
    # Uniform text -> we cast dtypes ourselves (NUMERIC/TIMESTAMP arrive as text).
    "Neon-Raw-Text-Output": "true",
}


def run_sql(query: str, params=None, timeout: int = 60):
    """Run one SQL statement against Neon via the HTTP SQL API; return parsed JSON."""
    body = {"query": query}
    if params is not None:
        body["params"] = params
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = _SESSION.post(SQL_ENDPOINT, headers=_HEADERS, json=body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:  # retryable
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(2 ** attempt * 0.4, 5))
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Neon HTTP SQL API error (HTTP {r.status_code}): {r.text[:500]}"
                )
            return r.json()
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt * 0.4, 5))
    raise RuntimeError(f"Neon SQL request failed after {MAX_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def fmt(ts) -> str:
    """Format a timestamp as Neon/Postgres-friendly text."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def to_df(rows) -> pd.DataFrame:
    """Convert Neon rows (all text) to a typed, sorted DataFrame."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    df["meter_id"] = df["meter_id"].astype(str)
    df["building_id"] = df["building_id"].astype(str)
    return df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


def fetch_january() -> pd.DataFrame:
    """Load only 1-31 January 2016 for all three meters."""
    rows = run_sql(
        f"SELECT timestamp, meter_id, building_id, electricity_kwh "
        f"FROM {TABLE} WHERE timestamp >= $1 AND timestamp <= $2 "
        f"ORDER BY meter_id, timestamp",
        params=[JAN_START, JAN_END],
    )["rows"]
    return to_df(rows)


def fetch_dataset_max():
    """Return the latest timestamp in the dataset (end of the replay)."""
    t = run_sql(f"SELECT MAX(timestamp) AS t FROM {TABLE}")["rows"][0]["t"]
    if t is None or (isinstance(t, str) and not t.strip()):
        return None
    return pd.to_datetime(t)


def fetch_next_hour(after_ts):
    """Fetch the next recorded hour after `after_ts` (one reading per meter).

    Returns a list of rows, or None when there are no more records.
    """
    nxt = run_sql(
        f"SELECT MIN(timestamp) AS t FROM {TABLE} WHERE timestamp > $1",
        params=[fmt(after_ts)],
    )["rows"]
    t = nxt[0]["t"] if nxt else None
    if t is None or (isinstance(t, str) and not t.strip()):
        return None
    rows = run_sql(
        f"SELECT timestamp, meter_id, building_id, electricity_kwh "
        f"FROM {TABLE} WHERE timestamp = $1 ORDER BY meter_id",
        params=[t],
    )["rows"]
    return rows


# ---------------------------------------------------------------------------
# Prediction helpers (saved model, no retraining)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    """Load the saved model and its feature order once per server process."""
    model = joblib.load(MODEL_PATH)
    feature_order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    return model, feature_order


def build_model_inputs(series: pd.Series, target_ts: pd.Timestamp, feature_order):
    """Build one meter's model inputs for `target_ts` from its revealed readings.

    `series` holds that meter's kWh readings indexed by timestamp. Each lag is
    looked up by exact timestamp (target minus N hours), so gaps are detected.
    Returns (values_dict, missing_list). `missing_list` has the feature names
    (with the timestamp each one needed) that could not be built.
    """
    values, missing = {}, []
    for name in feature_order:
        if name == "hour_of_day":
            values[name] = int(target_ts.hour)
        elif name == "day_of_week":
            values[name] = int(target_ts.dayofweek)  # Monday=0 .. Sunday=6
        elif name in LAG_HOURS:
            needed = target_ts - pd.Timedelta(hours=LAG_HOURS[name])
            if needed in series.index:
                values[name] = float(series.loc[needed])
            else:
                missing.append(f"{name} (reading at {fmt(needed)})")
        else:
            missing.append(f"{name} (unknown feature)")
    return values, missing


def predict_next_hour(df: pd.DataFrame, model, feature_order):
    """Predict the next hour for each meter using only the readings in `df`.

    The target for each meter is one hour after that meter's latest revealed
    reading. Returns a list of per-meter result dicts.
    """
    results = []
    for meter in sorted(df["meter_id"].unique()):
        series = (
            df[df["meter_id"] == meter]
            .drop_duplicates("timestamp")
            .set_index("timestamp")["electricity_kwh"]
            .sort_index()
        )
        target_ts = series.index.max() + pd.Timedelta(hours=1)
        values, missing = build_model_inputs(series, target_ts, feature_order)
        if missing:
            results.append({
                "meter": meter, "target": target_ts, "kwh": None,
                "message": "Not enough history to predict. Missing: " + "; ".join(missing),
            })
            continue
        X = pd.DataFrame([values], columns=feature_order)
        results.append({
            "meter": meter, "target": target_ts,
            "kwh": float(model.predict(X)[0]), "message": None,
        })
    return results


# ---------------------------------------------------------------------------
# Simulation state (session_state)
# ---------------------------------------------------------------------------
def reset_state_to_january():
    """(Re)initialise the simulation to the January starting data."""
    df = fetch_january()
    st.session_state.display_df = df
    st.session_state.current_max_ts = df["timestamp"].max()
    st.session_state.running = True
    st.session_state.finished = False
    st.session_state.last_tick = time.time()
    st.session_state.tick_count = 0
    st.session_state.prediction = None  # clear any earlier prediction
    if "dataset_max_ts" not in st.session_state or st.session_state.dataset_max_ts is None:
        st.session_state.dataset_max_ts = fetch_dataset_max()


if "display_df" not in st.session_state:
    reset_state_to_january()
st.session_state.setdefault("prediction", None)


def advance_one_hour():
    """Fetch the next recorded hour from Neon and append it; stop if at the end."""
    rows = fetch_next_hour(st.session_state.current_max_ts)
    if rows:
        new_df = to_df(rows)
        st.session_state.display_df = pd.concat(
            [st.session_state.display_df, new_df], ignore_index=True
        )
        st.session_state.current_max_ts = st.session_state.display_df["timestamp"].max()
        st.session_state.tick_count += 1
        end = st.session_state.get("dataset_max_ts")
        if end is not None and st.session_state.current_max_ts >= end:
            st.session_state.finished = True
            st.session_state.running = False
    else:
        # No more records in the dataset -> stop refreshing.
        st.session_state.finished = True
        st.session_state.running = False


@st.fragment(run_every=timedelta(seconds=INTERVAL_SEC))
def simulation_tick():
    """Fires every INTERVAL_SEC (and on app reruns). Advances once per real minute.

    The time guard ensures we only fetch a new hour on the ~minute tick, not on
    ordinary reruns (e.g. a button press). When it advances, it reruns the whole
    app so the charts update.
    """
    now = time.time()
    last = st.session_state.get("last_tick", 0.0)
    if (st.session_state.running and not st.session_state.finished
            and (now - last) >= INTERVAL_SEC):
        advance_one_hour()
        st.session_state.last_tick = now
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ FM Energy Dashboard")
    st.caption("Historical data · simulated live replay")
    st.divider()
    st.markdown(f"**Source:** Neon Postgres")
    st.markdown(f"**Table:** `{TABLE}`")
    st.markdown(f"**Host:** `{NEON_HOST}`")
    st.caption("Read via the Neon HTTP SQL API over HTTPS.")
    st.divider()
    st.markdown(f"**Replay window:** {JAN_START} → {fmt(st.session_state.get('dataset_max_ts'))}")
    st.caption("A new recorded hour is fetched every real minute.")

# ---------------------------------------------------------------------------
# Header + simulation controls
# ---------------------------------------------------------------------------
st.title("⚡ FM Energy Dashboard")
badge = (
    "<span style='background:#eef2ff;color:#3730a3;padding:2px 10px;"
    "border-radius:999px;font-size:0.78em;font-weight:700;letter-spacing:0.04em'>"
    "HISTORICAL DATA · SIMULATED LIVE</span>"
)
st.markdown(
    f"{badge} &nbsp; Hourly electricity for three meters — replaying recorded readings "
    f"from Neon, one new hour every real minute.",
    unsafe_allow_html=True,
)

cA, cB, cC, cD = st.columns([1.1, 1, 1, 1.2])
cA.metric("Simulated time", fmt(st.session_state.current_max_ts))
cB.metric("Hours replayed", f"{st.session_state.tick_count}")
cC.metric("Records shown", f"{len(st.session_state.display_df):,}")
status = "Completed" if st.session_state.finished else ("Running" if st.session_state.running else "Stopped")
cD.metric("Status", status)

cc1, cc2 = st.columns([3, 1])
with cc1:
    if st.session_state.finished:
        st.success("Simulation complete — reached the end of the dataset. Refreshing stopped.")
    else:
        st.info(f"Replaying… the next recorded hour is fetched every {INTERVAL_SEC}s. "
                f"Started from {JAN_START}.")
with cc2:
    if st.button("🔁 Restart simulation", type="primary"):
        reset_state_to_january()

# ---------------------------------------------------------------------------
# Charts (rendered from session_state.display_df, updated each rerun)
# ---------------------------------------------------------------------------
df = st.session_state.display_df
meters = sorted(df["meter_id"].unique().tolist())

METER_COLORS = {"METER_01": "#1f77b4", "METER_02": "#ff7f0e", "METER_03": "#2ca02c"}

st.divider()
st.subheader("Hourly electricity by meter")

cols = st.columns(3)
for col, meter in zip(cols, meters):
    mdf = df[df["meter_id"] == meter].sort_values("timestamp")
    building = mdf["building_id"].iloc[0] if len(mdf) else ""
    fig = px.line(mdf, x="timestamp", y="electricity_kwh", title=meter)
    fig.update_traces(
        line_color=METER_COLORS.get(meter, "#1f77b4"),
        hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>kWh: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white", height=340, margin=dict(l=20, r=20, t=45, b=20),
        title_font_size=15, showlegend=False,
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Electricity (kWh)")
    with col:
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{building} · {len(mdf):,} hourly readings")

st.divider()
st.subheader("Total electricity usage (all meters, summed by timestamp)")

total = df.groupby("timestamp", as_index=False)["electricity_kwh"].sum()
fig2 = px.line(total, x="timestamp", y="electricity_kwh")
fig2.update_traces(
    line_color="#6366f1", fill="tozeroy",
    hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Total kWh: %{y:.2f}<extra></extra>",
)
fig2.update_layout(
    template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
)
fig2.update_xaxes(title_text="Time", rangeslider_visible=True)
fig2.update_yaxes(title_text="Electricity (kWh)")
st.plotly_chart(fig2, width="stretch")
st.caption("Sum of METER_01 + METER_02 + METER_03 at each timestamp (kWh).")

# ---------------------------------------------------------------------------
# Prediction (new in v3) and LLM Analysis placeholder
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Prediction and analysis")

p1, p2 = st.columns(2)
with p1:
    with st.container(border=True):
        st.markdown("#### 🔮 Prediction")
        st.caption("Saved model `models/gbr_model.joblib` (not retrained). Each meter "
                   "is predicted from its own readings revealed so far.")
        if st.button("Predict next hour", key="predict_btn"):
            if st.session_state.finished:
                st.session_state.prediction = {
                    "error": "The simulation reached the last record, so there is no "
                             "next recorded hour to predict. Press Restart simulation.",
                }
            else:
                try:
                    model, feature_order = load_model_and_features()
                    st.session_state.prediction = {
                        "as_of": st.session_state.current_max_ts,
                        "results": predict_next_hour(
                            st.session_state.display_df, model, feature_order),
                    }
                except Exception as e:  # noqa: BLE001 - show a clear message
                    st.session_state.prediction = {
                        "error": f"Could not run the saved model: {e}",
                    }

        pred = st.session_state.prediction
        if pred is None:
            st.info("Press **Predict next hour** to forecast each meter's next recorded hour.")
        elif "error" in pred:
            st.warning(pred["error"])
        else:
            st.caption(f"Based on readings revealed up to {fmt(pred['as_of'])}.")
            pcols = st.columns(len(pred["results"]))
            for pcol, res in zip(pcols, pred["results"]):
                with pcol:
                    if res["kwh"] is None:
                        st.markdown(f"**{res['meter']}**")
                        st.caption(f"Target time: {fmt(res['target'])}")
                        st.warning(res["message"])
                    else:
                        st.metric(
                            label=f"{res['meter']} · {fmt(res['target'])}",
                            value=f"{res['kwh']:.2f} kWh",
                        )
            st.caption("The saved model was trained on one meter averaging about "
                       "26 kWh, so it is less reliable for meters with much higher usage.")
with p2:
    with st.container(border=True):
        st.markdown("#### 🤖 LLM Analysis")
        st.info("LLM-powered energy usage analysis will appear here.")

# ---------------------------------------------------------------------------
# Register the one-minute ticker (only while the simulation is active).
# ---------------------------------------------------------------------------
if st.session_state.running and not st.session_state.finished:
    simulation_tick()
