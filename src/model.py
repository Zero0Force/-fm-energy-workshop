"""Model training and evaluation helpers for the FM Energy Workshop."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42


def train_model(X_train, y_train):
    """Train a simple boosted-tree regressor on the training data.

    Uses scikit-learn's GradientBoostingRegressor with random_state=42 and
    otherwise default settings (kept simple, as specified).
    """
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """Return model predictions for feature matrix X."""
    return model.predict(X)


def evaluate(y_true, y_pred):
    """Return (MAE, RMSE) in the target's units (kWh)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def baseline_previous_hour(X):
    """Baseline prediction: use the previous hour's consumption (the lag_1 feature)."""
    return X["lag_1"].to_numpy()
