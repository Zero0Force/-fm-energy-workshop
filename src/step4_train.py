"""Step 4 - Train a simple boosted-tree model.

Trains scikit-learn GradientBoostingRegressor (random_state=42) on the
training set. Test-set evaluation comes in Step 5; here we only confirm
the model trained and inspect what it learned from the training data.
"""
from pathlib import Path
import pandas as pd
from prepare import load_prepared, split_train_test, make_xy, FEATURES, TARGET
from model import train_model, RANDOM_STATE

REPORT_PATH = Path(r"c:\Users\user\workshop\_step4_out.txt")

lines = []


def w(s=""):
    lines.append(str(s))


data = load_prepared()
train, test = split_train_test(data)
X_train, y_train = make_xy(train)

model = train_model(X_train, y_train)

w("STEP 4 - TRAIN A SIMPLE BOOSTED-TREE MODEL")
w("=" * 60)
w(f"Model: {type(model).__name__}")
w(f"random_state: {RANDOM_STATE}")
w("")
w("Key hyperparameters (defaults unless noted):")
params = model.get_params()
for k in ["loss", "learning_rate", "n_estimators", "max_depth",
          "min_samples_split", "min_samples_leaf", "subsample"]:
    w(f"  {k}: {params[k]}")
w("")

# Did it learn anything from the training data? (in-sample fit only, not the test eval)
train_r2 = model.score(X_train, y_train)
w("Training-set fit (IN-SAMPLE, not the test evaluation):")
w(f"  R^2 on training data: {train_r2:.4f}")
w("  (close to 1 means it fits the training data well; the real test is Step 5.)")
w("")

# Which features mattered most to the model?
w("Feature importances (what the model relied on):")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for name, val in imp.items():
    w(f"  {name:<14} {val:.4f}")
w(f"  sum: {imp.sum():.4f}")
w("")

w(f"Number of boosted trees (estimators): {model.n_estimators}")
w(f"Training rows: {len(X_train)} | features: {len(FEATURES)}")
w("")
w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"Trained {type(model).__name__} | train R^2={train_r2:.4f}")
print("DONE")
