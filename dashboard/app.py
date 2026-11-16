"""FM Energy Workshop - historical electricity dashboard.

Reads the `meter_readings_source` table from Neon Postgres (over the Neon HTTP
SQL API, using DATABASE_URL from .env) and shows:
  - three per-meter hourly electricity charts across the top
  - a total chart (all three meters summed by timestamp, in kWh) below them
  - placeholders for Prediction and LLM Analysis at the bottom

The data is historical (not live). No replay or prediction yet.

Run from the project root (fm-energy-workshop) with the `vibe` venv:
    c:\\Users\\user\\workshop\\vibe\\Scripts\\python.exe -m streamlit run dashboard/app.py --server.headless true
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

# ---------------------------------------------------------------------------
# Page config - must be the first Streamlit command.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & Neon connection setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # fm-energy-workshop
ENV_PATH = PROJECT_ROOT / ".env"
TABLE = "meter_readings_source"
MAX_RETRIES = 4

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
    # Ask Neon to return every value as text so dtypes are uniform; we cast them
    # ourselves below. (NUMERIC and TIMESTAMP otherwise arrive as strings anyway.)
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


@st.cache_data(ttl=600, show_spinner="Loading historical readings from Neon…")
def load_readings() -> pd.DataFrame:
    """Load all rows from meter_readings_source as a tidy DataFrame."""
    rows = run_sql(
        f"SELECT timestamp, meter_id, building_id, electricity_kwh "
        f"FROM {TABLE} ORDER BY meter_id, timestamp"
    )["rows"]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    df["meter_id"] = df["meter_id"].astype(str)
    df["building_id"] = df["building_id"].astype(str)
    return df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df = load_readings()
except Exception as e:  # noqa: BLE001 - surface a friendly error in the app
    st.error(f"Could not load readings from Neon (`{TABLE}`):\n\n{e}")
    st.stop()

meters = sorted(df["meter_id"].unique().tolist())
n_rows = len(df)
t_min = df["timestamp"].min()
t_max = df["timestamp"].max()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ FM Energy Dashboard")
    st.caption("Historical data")
    st.divider()
    st.markdown(f"**Source:** Neon Postgres")
    st.markdown(f"**Table:** `{TABLE}`")
    st.markdown(f"**Host:** `{NEON_HOST}`")
    st.markdown(f"**Meters:** {', '.join(meters)}")
    st.caption("Read via the Neon HTTP SQL API over HTTPS.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ FM Energy Dashboard")
badge = (
    "<span style='background:#eef2ff;color:#3730a3;padding:2px 10px;"
    "border-radius:999px;font-size:0.78em;font-weight:700;letter-spacing:0.04em'>"
    "HISTORICAL DATA</span>"
)
st.markdown(
    f"{badge} &nbsp; Hourly electricity usage · three meters · from Neon Postgres",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Hourly readings", f"{n_rows:,}")
c2.metric("Meters", len(meters))
c3.metric("Date range", f"{t_min:%Y-%m-%d} → {t_max:%Y-%m-%d}")

# ---------------------------------------------------------------------------
# Per-meter charts (side by side)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Hourly electricity by meter")

METER_COLORS = {
    "METER_01": "#1f77b4",
    "METER_02": "#ff7f0e",
    "METER_03": "#2ca02c",
}

cols = st.columns(3)
for col, meter in zip(cols, meters):
    mdf = df[df["meter_id"] == meter].sort_values("timestamp")
    building = mdf["building_id"].iloc[0]

    fig = px.line(mdf, x="timestamp", y="electricity_kwh", title=meter)
    fig.update_traces(
        line_color=METER_COLORS.get(meter, "#1f77b4"),
        hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>kWh: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        title_font_size=15,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Electricity (kWh)")

    with col:
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{building} · {len(mdf):,} hourly readings")

# ---------------------------------------------------------------------------
# Total chart (summed by timestamp)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Total electricity usage (all meters, summed by timestamp)")

total = df.groupby("timestamp", as_index=False)["electricity_kwh"].sum()
fig2 = px.line(total, x="timestamp", y="electricity_kwh")
fig2.update_traces(
    line_color="#6366f1",
    fill="tozeroy",
    hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Total kWh: %{y:.2f}<extra></extra>",
)
fig2.update_layout(
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
)
fig2.update_xaxes(title_text="Time", rangeslider_visible=True)
fig2.update_yaxes(title_text="Electricity (kWh)")
st.plotly_chart(fig2, width="stretch")
st.caption("Sum of METER_01 + METER_02 + METER_03 at each timestamp (kWh).")

# ---------------------------------------------------------------------------
# Placeholders: Prediction & LLM Analysis
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Coming next")

p1, p2 = st.columns(2)
with p1:
    with st.container(border=True):
        st.markdown("#### 🔮 Prediction")
        st.info("Next-hour electricity forecast will appear here.")
with p2:
    with st.container(border=True):
        st.markdown("#### 🤖 LLM Analysis")
        st.info("LLM-powered energy usage analysis will appear here.")
