from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prepare_kaggle_data import OUTPUT_FILE, prepare_dataset


MODEL_PATH = Path("models/random_forest_model.pkl")
FORECAST_HORIZON_DAYS = 7

FEATURES = [
    "warehouse_code",
    "product_code",
    "daily_sales",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_mean_7",
    "sales_mean_30",
    "day_of_week",
    "month",
    "is_weekend",
]
TARGET = "future_demand"


def load_training_data():
    if not OUTPUT_FILE.exists():
        prepare_dataset()

    df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
    missing = [
        column
        for column in FEATURES + [TARGET, "date"]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    df = df.sort_values("date").dropna(subset=FEATURES + [TARGET])
    return df


def train_model():
    df = load_training_data()

    unique_dates = np.sort(df["date"].unique())
    cutoff_index = int(len(unique_dates) * 0.80)
    cutoff_date = unique_dates[cutoff_index]

    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    print("[AI] Training Random Forest...")
    print(f"[AI] Forecast horizon: {FORECAST_HORIZON_DAYS} days")
    print(f"[AI] Train rows: {len(train_df):,}")
    print(f"[AI] Test rows: {len(test_df):,}")
    print(f"[AI] Time cutoff: {pd.Timestamp(cutoff_date).date()}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    print("\n===== MODEL EVALUATION =====")
    print(f"MAE  : {metrics['mae']:.2f}")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"R²   : {metrics['r2']:.4f}")

    print("\n===== FEATURE IMPORTANCE =====")
    for feature, score in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature:<22} {score:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "forecast_horizon_days": FORECAST_HORIZON_DAYS,
            "metrics": metrics,
            "source": (
                "Kaggle: talhanazir168/"
                "store-inventory-demand-forecasting-dataset"
            ),
        },
        MODEL_PATH,
    )

    print(f"\n[AI] Model saved: {MODEL_PATH}")
    return model, metrics


if __name__ == "__main__":
    train_model()
