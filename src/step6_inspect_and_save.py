"""Step 6 - Inspect example predictions and save the work.

Plots actual vs. predicted over the two-week test window, then saves the
fitted model to models/gbr_model.joblib and the feature order to
models/feature_order.json for the later dashboard.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: save plots without a display
import matplotlib.pyplot as plt
import joblib

from prepare import load_prepared, split_train_test, make_xy, FEATURES, TARGET, REPO_ROOT
from model import train_model, predict, evaluate, baseline_previous_hour

REPORT_PATH = Path(r"c:\Users\user\workshop\_step6_out.txt")
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
PLOT_PATH = REPORTS_DIR / "actual_vs_predicted.png"
MODEL_PATH = MODELS_DIR / "gbr_model.joblib"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"

lines = []


def w(s=""):
    lines.append(str(s))


# Reproduce the trained model and its test predictions
data = load_prepared()
train, test = split_train_test(data)
X_train, y_train = make_xy(train)
X_test, y_test = make_xy(test)
model = train_model(X_train, y_train)

model_pred = predict(model, X_test)
base_pred = baseline_previous_hour(X_test)
y = y_test.to_numpy()
model_mae, model_rmse = evaluate(y, model_pred)
base_mae, base_rmse = evaluate(y, base_pred)

# 1) Plot actual vs predicted over the two-week test window (baseline shown for context)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test.index, y, label="actual", color="black", linewidth=1.4)
ax.plot(test.index, model_pred, label="predicted (model)", color="#1f77b4", linewidth=1.2)
ax.plot(test.index, base_pred, label="baseline (previous hour)",
        color="#d62728", linewidth=1.0, linestyle="--", alpha=0.7)
ax.set_title(
    f"FM Energy Workshop - next-hour forecast vs actual\n"
    f"Test window {test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d}  "
    f"(model MAE={model_mae:.2f} vs baseline MAE={base_mae:.2f} kWh)"
)
ax.set_xlabel("time")
ax.set_ylabel("electricity (kWh)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=130)
plt.close(fig)

# 2) Save the fitted model and the feature order
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
FEATURE_ORDER_PATH.write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")

# Report
w("STEP 6 - INSPECT EXAMPLE PREDICTIONS AND SAVE THE WORK")
w("=" * 60)
w("Recap of metrics on the two-week test window:")
w(f"  Model     MAE={model_mae:.3f}  RMSE={model_rmse:.3f} kWh")
w(f"  Baseline  MAE={base_mae:.3f}  RMSE={base_rmse:.3f} kWh")
w("")
w("Saved artifacts:")
w(f"  model        : {MODEL_PATH.relative_to(REPO_ROOT)}  ({MODEL_PATH.stat().st_size} bytes)")
w(f"  feature order: {FEATURE_ORDER_PATH.relative_to(REPO_ROOT)}  ({FEATURE_ORDER_PATH.read_text().strip()})")
w(f"  plot         : {PLOT_PATH.relative_to(REPO_ROOT)}  ({PLOT_PATH.stat().st_size} bytes)")
w("")
w("Feature order the dashboard must use (exact order):")
w("  " + str(FEATURES))
w("")
w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"Saved model -> {MODEL_PATH}")
print(f"Saved feature order -> {FEATURE_ORDER_PATH}")
print(f"Saved plot -> {PLOT_PATH}")
print("DONE")
