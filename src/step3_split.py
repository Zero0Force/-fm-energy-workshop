"""Step 3 - Split into train and test sets.

Hold back the last two weeks (336 hours) as a test set the model never
sees during training. Train on everything before that. The split is by
time, not random, so the future stays unseen.
"""
from pathlib import Path
from prepare import load_prepared, split_train_test, make_xy, FEATURES, TARGET, TEST_HOURS

REPORT_PATH = Path(r"c:\Users\user\workshop\_step3_out.txt")

lines = []


def w(s=""):
    lines.append(str(s))


data = load_prepared()
train, test = split_train_test(data)
X_train, y_train = make_xy(train)
X_test, y_test = make_xy(test)

w("STEP 3 - SPLIT INTO TRAIN AND TEST SETS")
w("=" * 60)
w(f"Total prepared rows: {len(data)}")
w(f"Test set = last {TEST_HOURS} hours ({TEST_HOURS // 24} days = {TEST_HOURS // (24 * 7)} weeks); train = the rest")
w("")

w("TRAIN")
w(f"  rows: {len(train)}")
w(f"  from: {train.index.min()}")
w(f"  to:   {train.index.max()}")
w("")

w("TEST")
w(f"  rows: {len(test)}")
w(f"  from: {test.index.min()}")
w(f"  to:   {test.index.max()}")
w("")

w("CHECKS")
w("-" * 60)
gap = test.index.min() - train.index.max()
w(f"train ends at: {train.index.max()}")
w(f"test  starts at: {test.index.min()}")
w(f"gap between train end and test start: {gap}  (1 hour = contiguous, no overlap)")
overlap = set(train.index) & set(test.index)
w(f"overlapping timestamps: {len(overlap)}  (0 = good)")
w(f"train + test rows: {len(train) + len(test)} (should equal {len(data)})")
w(f"test is exactly {TEST_HOURS // 24} days = {TEST_HOURS} hours: {len(test) == TEST_HOURS}")
w("")

w("Shapes:")
w(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
w(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
w(f"  features used: {list(X_train.columns)}")
w(f"  target: {TARGET}")
w("")

w("First row the model must predict (start of the test window):")
w(f"  timestamp: {test.index[0]}")
w(f"  actual target (y): {float(y_test.iloc[0])}")
w(f"  features (X): {X_test.iloc[0].to_dict()}")
w("")

w("DONE")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Report written to", REPORT_PATH)
print(f"train: {len(train)} ({train.index.min()} .. {train.index.max()})")
print(f"test:  {len(test)} ({test.index.min()} .. {test.index.max()})")
print("DONE")
