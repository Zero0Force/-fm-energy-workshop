"""Step 1 - Understand the data.

Load Data/one_meter.csv, describe it briefly, and flag anything that
needs cleaning or preprocessing. This step is read-only: it does not
modify the CSV.
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "Data" / "one_meter.csv"
REPORT_PATH = Path(r"c:\Users\user\workshop\_step1_out.txt")

lines = []

def w(s=""):
    lines.append(str(s))


# 1. Load raw file
df = pd.read_csv(CSV_PATH)
w("STEP 1 - UNDERSTAND THE DATA")
w("=" * 60)
w(f"Loaded: {CSV_PATH.name}")
w(f"Shape (rows, cols): {df.shape}")
w("")
w("Columns and dtypes:")
w(df.dtypes.to_string())
w("")
w("First 5 rows:")
w(df.head().to_string())
w("")
w("Last 5 rows:")
w(df.tail().to_string())
w("")

# 2. Identifiers (is this a single series or many?)
w(f"Unique meter_id:    {df['meter_id'].unique().tolist()}")
w(f"Unique building_id: {df['building_id'].unique().tolist()}")
w("")

# 3. Parse timestamp, sort, set as time index
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
dups = int(df["timestamp"].duplicated().sum())
w(f"Duplicate timestamps: {dups}")
df = df.set_index("timestamp")

w("")
w("Time index range:")
w(f"  min: {df.index.min()}")
w(f"  max: {df.index.max()}")
w(f"  count: {len(df)}")
expected = int((df.index.max() - df.index.min()).total_seconds() / 3600) + 1
w(f"  expected if perfectly hourly: {expected}")
w("")

# 4. Missing values
w("Missing values per column:")
w(df.isna().sum().to_string())
w("")

# 5. Time gaps (anything that is not exactly 1 hour)
diffs = df.index.to_series().diff().dropna()
gaps = diffs[diffs != pd.Timedelta(hours=1)]
w(f"Time steps not equal to 1 hour: {len(gaps)}")
if len(gaps):
    w(gaps.head(10).to_string())
w("")

# 6. Summary stats of the target column
w("electricity_kwh summary:")
w(df["electricity_kwh"].describe().round(2).to_string())
w("")

# 7. Cleaning / preprocessing assessment
w("=" * 60)
w("CLEANING / PREPROCESSING NEEDS")
w("-" * 60)
needs = []
if df["meter_id"].nunique() == 1 and df["building_id"].nunique() == 1:
    needs.append("Single meter and building: meter_id and building_id are constant, "
                 "so model only needs timestamp + electricity_kwh.")
if dups == 0:
    needs.append("No duplicate timestamps: no de-duplication needed.")
else:
    needs.append("Duplicate timestamps found: decide a keep/aggregate rule.")
if df.isna().sum().sum() == 0:
    needs.append("No missing values: no imputation needed.")
else:
    needs.append("Missing values present: impute or drop per a defined rule.")
if len(gaps) == 0 and expected == len(df):
    needs.append("Perfectly hourly with no gaps: no reindex/forward-fill needed.")
else:
    needs.append("Gaps in time: reindex to an hourly grid and impute missing hours.")
needs.append("Parse timestamp to datetime and use as the time index (done in this step).")
needs.append("No scaling/encoding needed for the tree model used in later steps.")
for n in needs:
    w(" - " + n)

w("")
w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"Rows: {len(df)} | Range: {df.index.min()} .. {df.index.max()}")
print("DONE")
