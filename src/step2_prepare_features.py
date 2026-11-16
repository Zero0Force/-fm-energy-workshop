"""Step 2 - Prepare the data.

Builds the features used to predict electricity_kwh at hour t and drops
the first 168 rows that lack full history. Read-only on the raw CSV.
"""
from pathlib import Path
import pandas as pd
from prepare import load_series, build_features, FEATURES, TARGET, DROP_FIRST

REPORT_PATH = Path(r"c:\Users\user\workshop\_step2_out.txt")

lines = []


def w(s=""):
    lines.append(str(s))


ser = load_series()
w("STEP 2 - PREPARE THE DATA")
w("=" * 60)
w(f"Rows before feature build: {len(ser)}")
w("")

data = build_features(ser)

w("Feature columns and dtypes (after dropping first 168):")
w(data[FEATURES + [TARGET]].dtypes.to_string())
w("")
w(f"Rows after dropping first {DROP_FIRST}: {len(data)}")
w("")

# Checks
w("CHECKS")
w("-" * 60)
w(f"Final shape (rows, cols): {data.shape}")
w(f"Feature order: {FEATURES}")
w(f"Target: {TARGET}")
w(f"Any NaN remaining? {int(data.isna().sum().sum())}  (0 = good)")
w(f"hour_of_day range: {int(data['hour_of_day'].min())} .. {int(data['hour_of_day'].max())}")
w(f"day_of_week range: {int(data['day_of_week'].min())} .. {int(data['day_of_week'].max())}")
w("")

# Sanity check on lags at the first kept row
first_ts = data.index[0]
ref168 = ser.loc[first_ts - pd.Timedelta(hours=168)]
ref1 = ser.loc[first_ts - pd.Timedelta(hours=1)]
w("Sanity check (first kept row):")
w(f"  first kept timestamp: {first_ts}")
w(f"  lag_168 should equal y[{first_ts - pd.Timedelta(hours=168)}] = {ref168}")
w(f"  actual lag_168: {float(data['lag_168'].iloc[0])}")
w(f"  lag_1 should equal y[{first_ts - pd.Timedelta(hours=1)}] = {ref1}")
w(f"  actual lag_1:   {float(data['lag_1'].iloc[0])}")
w("")

w("First 5 rows:")
w(data[FEATURES + [TARGET]].head().to_string())
w("")
w("Last 5 rows:")
w(data[FEATURES + [TARGET]].tail().to_string())
w("")
w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"Prepared rows: {len(data)} | features: {FEATURES}")
print("DONE")
