#!/usr/bin/env python3
"""Load Data/three_meters.csv into Neon Postgres via the Neon HTTP SQL API.

Why HTTP and not psycopg2: port 5432 may be blocked by the local firewall, so we
talk to Neon over HTTPS at https://<host>/sql using the `requests` library. The
Postgres connection string is passed in the `Neon-Connection-String` request
header (channel_binding=require is stripped first).

Run with the pre-created `vibe` venv:
    c:\\Users\\user\\workshop\\vibe\\Scripts\\python.exe src\\load_to_neon.py
"""
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../fm-energy-workshop/src
PROJECT_DIR = HERE.parent                        # .../fm-energy-workshop
ENV_PATH = PROJECT_DIR / ".env"
CSV_PATH = PROJECT_DIR / "Data" / "three_meters.csv"

TABLE = "meter_readings_source"
COLUMNS = ["timestamp", "meter_id", "building_id", "electricity_kwh"]
BATCH_SIZE = 500        # rows per multi-row INSERT request
MAX_RETRIES = 4         # per request, for transient 429/5xx/network errors


# ----------------------------------------------------------------------------
# Connection setup
# ----------------------------------------------------------------------------
def strip_channel_binding(url: str) -> str:
    """Remove channel_binding=require and tidy any leftover separators."""
    for frag in ("&channel_binding=require", "channel_binding=require&",
                 "channel_binding=require"):
        url = url.replace(frag, "")
    # clean dangling separators left behind
    url = url.replace("?&", "?").replace("&&", "&")
    if url.endswith("&"):
        url = url[:-1]
    if url.endswith("?"):
        url = url[:-1]
    return url


def redact(url: str) -> str:
    """Connection string with the password masked, for safe logging."""
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.username:
        netloc = f"{p.username}:***@{p.hostname}"
        if p.port:
            netloc += f":{p.port}"
    out = f"{p.scheme}://{netloc}{p.path}"
    if p.query:
        out += f"?{p.query}"
    return out


load_dotenv(ENV_PATH)
RAW_DB_URL = os.environ.get("DATABASE_URL")
if not RAW_DB_URL:
    print(f"ERROR: DATABASE_URL not found in {ENV_PATH}", file=sys.stderr)
    sys.exit(1)

DB_URL = strip_channel_binding(RAW_DB_URL)
NEON_HOST = urlparse(DB_URL).hostname
SQL_ENDPOINT = f"https://{NEON_HOST}/sql"

SESSION = requests.Session()
BASE_HEADERS = {
    "Neon-Connection-String": DB_URL,
    "Content-Type": "application/json",
}


def run_sql(query: str, params=None, raw_text: bool = False, timeout: int = 120):
    """Execute one SQL statement via the Neon HTTP SQL API; return parsed JSON."""
    headers = dict(BASE_HEADERS)
    if raw_text:
        headers["Neon-Raw-Text-Output"] = "true"
    body = {"query": query}
    if params is not None:
        body["params"] = params

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.post(SQL_ENDPOINT, headers=headers, json=body, timeout=timeout)
            # retryable: rate limit or server error
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(2 ** attempt * 0.4, 5))
                continue
            # hard failure
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Neon HTTP SQL API returned HTTP {r.status_code}.\n"
                    f"Response: {r.text[:800]}"
                )
            return r.json()
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt * 0.4, 5))
    raise RuntimeError(f"Neon SQL request failed after {MAX_RETRIES} attempts: {last_err}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Load three_meters.csv -> Neon (HTTP SQL API)")
    print("=" * 78)
    print(f"Neon host          : {NEON_HOST}")
    print(f"SQL endpoint       : {SQL_ENDPOINT}")
    print(f"Connection string  : {redact(DB_URL)}")
    print(f"channel_binding     : stripped (was present: "
          f"{'channel_binding=require' in RAW_DB_URL})")

    # [1] connectivity probe -------------------------------------------------
    print("\n[1/5] Connectivity probe: SELECT 1 ...")
    probe = run_sql("SELECT 1 AS one")
    print("    -> ok, rows:", probe.get("rows"))

    # [2] (re)create table --------------------------------------------------
    print(f"\n[2/5] (Re)creating table {TABLE} ...")
    run_sql(f"DROP TABLE IF EXISTS {TABLE}")
    run_sql(
        f"""CREATE TABLE {TABLE} (
            timestamp        TIMESTAMP NOT NULL,
            meter_id         TEXT NOT NULL,
            building_id      TEXT NOT NULL,
            electricity_kwh  NUMERIC NOT NULL,
            CONSTRAINT meter_readings_source_meter_ts_key UNIQUE (meter_id, timestamp)
        )"""
    )
    print(f"    -> table ready; UNIQUE(meter_id, timestamp) enforced")

    # [3] read + validate CSV ----------------------------------------------
    print(f"\n[3/5] Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, dtype=str)   # keep raw text (preserves decimals)
    print(f"    -> rows={len(df)}, columns={list(df.columns)}")

    bad_ts = int(pd.to_datetime(df["timestamp"], errors="coerce").isna().sum())
    bad_kw = int(pd.to_numeric(df["electricity_kwh"], errors="coerce").isna().sum())
    dup_csv = int(df.duplicated(subset=["meter_id", "timestamp"]).sum())
    print(f"    -> invalid timestamps={bad_ts}, invalid kWh={bad_kw}, "
          f"duplicate (meter_id,timestamp) pairs={dup_csv}")
    if bad_ts or bad_kw:
        sys.exit("ERROR: CSV contains invalid values; aborting before upload.")
    if dup_csv:
        sys.exit("ERROR: CSV has duplicate (meter_id,timestamp) pairs; "
                 "unique constraint would be violated.")

    # [4] upload in batches --------------------------------------------------
    print(f"\n[4/5] Uploading {len(df)} rows in batches of {BATCH_SIZE} ...")
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    total_inserted = 0
    for bi, i in enumerate(range(0, len(df), BATCH_SIZE), start=1):
        batch = df.iloc[i:i + BATCH_SIZE]
        values_sql = ",".join(
            f"(${k * 4 + 1},${k * 4 + 2},${k * 4 + 3},${k * 4 + 4})"
            for k in range(len(batch))
        )
        query = f"INSERT INTO {TABLE} ({','.join(COLUMNS)}) VALUES {values_sql}"
        params = []
        for _, row in batch.iterrows():
            params.extend([row["timestamp"], row["meter_id"],
                           row["building_id"], row["electricity_kwh"]])
        res = run_sql(query, params=params)
        rc = res.get("rowCount")
        rc = rc if isinstance(rc, int) else len(batch)
        total_inserted += rc
        print(f"    -> batch {bi}/{n_batches}: inserted {rc} (running total {total_inserted})")
    print(f"    -> total inserted (API rowCount): {total_inserted}")

    # [5] verify + report ----------------------------------------------------
    print(f"\n[5/5] Verifying in Neon ...")
    # NOTE: COUNT(*) returns bigint (int8/OID 20). The Neon HTTP SQL proxy
    # maps int8 -> JSON string (only int2/int4 map to JSON number), so coerce.
    total_q = run_sql(f"SELECT COUNT(*) AS n FROM {TABLE}")
    db_total = int(total_q["rows"][0]["n"])
    print(f"    -> rows in {TABLE}: {db_total} (CSV had {len(df)})")

    dup_q = run_sql(
        f"SELECT COUNT(*) AS n FROM ("
        f"  SELECT meter_id, timestamp FROM {TABLE} "
        f"  GROUP BY meter_id, timestamp HAVING COUNT(*) > 1"
        f") d"
    )
    db_dup = int(dup_q["rows"][0]["n"])
    print(f"    -> duplicate (meter_id,timestamp) pairs in DB: {db_dup}")

    summary = run_sql(
        f"SELECT meter_id, COUNT(*) AS row_count, "
        f"MIN(timestamp) AS earliest, MAX(timestamp) AS latest "
        f"FROM {TABLE} GROUP BY meter_id ORDER BY meter_id",
        raw_text=True,
    )

    print("\n" + "=" * 78)
    print(f"{'meter_id':<12}{'row_count':>12}   {'earliest_timestamp':<22}"
          f"{'latest_timestamp':<22}")
    print("-" * 78)
    for r in summary["rows"]:
        print(f"{r['meter_id']:<12}{str(r['row_count']):>12}   "
              f"{str(r['earliest']):<22}{str(r['latest']):<22}")
    print("=" * 78)
    print(f"Total rows in {TABLE}: {db_total}")

    if db_total != len(df):
        sys.exit(f"ERROR: DB row count {db_total} != CSV row count {len(df)}.")
    if db_dup != 0:
        sys.exit(f"ERROR: {db_dup} duplicate (meter_id,timestamp) pairs found in DB.")
    print("\nUpload verified: row count matches CSV and (meter_id, timestamp) is unique.")


if __name__ == "__main__":
    main()
