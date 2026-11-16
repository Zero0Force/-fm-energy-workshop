"""Step 5 - Evaluate the model.

Compares the model's MAE and RMSE (in kWh) against a baseline that simply
uses the previous hour's consumption (the lag_1 feature), on the same
held-out test rows.
"""
from pathlib import Path
from prepare import load_prepared, split_train_test, make_xy, FEATURES, TARGET
from model import train_model, predict, evaluate, baseline_previous_hour

REPORT_PATH = Path(r"c:\Users\user\workshop\_step5_out.txt")

lines = []


def w(s=""):
    lines.append(str(s))


data = load_prepared()
train, test = split_train_test(data)
X_train, y_train = make_xy(train)
X_test, y_test = make_xy(test)

model = train_model(X_train, y_train)

# Model predictions on the test set
model_pred = predict(model, X_test)
# Baseline: previous hour's consumption (lag_1) for the same test rows
base_pred = baseline_previous_hour(X_test)

y = y_test.to_numpy()
model_mae, model_rmse = evaluate(y, model_pred)
base_mae, base_rmse = evaluate(y, base_pred)

w("STEP 5 - EVALUATE THE MODEL")
w("=" * 60)
w(f"Test window: {test.index.min()} .. {test.index.max()}  ({len(test)} hours)")
w("Baseline = previous hour's consumption (the lag_1 feature)")
w("")
w(f"{'Metric':<8} {'Model':>10} {'Baseline':>10} {'Better':>10}")
w("-" * 42)
better_mae = "Model" if model_mae < base_mae else "Baseline"
better_rmse = "Model" if model_rmse < base_rmse else "Baseline"
w(f"{'MAE':<8} {model_mae:>10.3f} {base_mae:>10.3f} {better_mae:>10}")
w(f"{'RMSE':<8} {model_rmse:>10.3f} {base_rmse:>10.3f} {better_rmse:>10}")
w("")
w("MAE  = mean absolute error in kWh (average size of misses)")
w("RMSE = root mean squared error in kWh (penalises big misses more)")
w("")

# Improvement over baseline
mae_impr = (base_mae - model_mae) / base_mae * 100
rmse_impr = (base_rmse - model_rmse) / base_rmse * 100
w(f"Model MAE  is {mae_impr:+.1f}% vs baseline ({'better' if mae_impr > 0 else 'worse'})")
w(f"Model RMSE is {rmse_impr:+.1f}% vs baseline ({'better' if rmse_impr > 0 else 'worse'})")
w("")

# A few example predictions vs actual vs baseline
w("Sample predictions (first 8 test rows):")
w(f"  {'timestamp':<19} {'actual':>7} {'model':>7} {'baseline':>9}")
for i in range(8):
    w(f"  {str(test.index[i]):<19} {float(y[i]):>7.2f} {float(model_pred[i]):>7.2f} {float(base_pred[i]):>9.2f}")
w("")
w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"Model  MAE={model_mae:.3f} RMSE={model_rmse:.3f}")
print(f"Base   MAE={base_mae:.3f} RMSE={base_rmse:.3f}")
print("DONE")
