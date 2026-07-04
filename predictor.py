import os
import re
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH = "models/random_forest_model.pkl"

model = None

FEATURE_COLUMNS = [
    "warehouse_code",
    "product_code",
    "inventory_quantity",
    "daily_sales",
    "incoming_stock",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_mean_7",
    "sales_mean_30",
    "day_of_week",
    "month",
    "is_weekend"
]

# =====================================
# LOAD MODEL
# =====================================

if os.path.exists(MODEL_PATH):

    try:

        data = joblib.load(MODEL_PATH)

        if isinstance(data, dict):

            model = data.get("model")

            FEATURE_COLUMNS = data.get(
                "features",
                FEATURE_COLUMNS
            )

        else:

            model = data

        print(
            "[AI] Model Loaded Successfully"
        )

    except Exception as e:

        print(
            "[AI] Model Load Error:",
            e
        )

        model = None

else:

    print(
        "[AI] Model Not Found -> Using Fallback"
    )


# =====================================
# SAFE NUMBER
# =====================================

def safe_number(value, default=0):

    try:

        if value is None:
            return float(default)

        if isinstance(value, dict):

            if "value" in value:

                value = value["value"]

            elif len(value) > 0:

                value = list(
                    value.values()
                )[0]

            else:

                return float(default)

        return float(value)

    except Exception:

        return float(default)


# =====================================
# FEATURE BUILDER
# =====================================

def build_features(
    inventory_quantity,
    order_quantity,
    daily_sales,
    incoming_stock,
    lead_time,
    delivery_status,
    vehicle_capacity,
    warehouse_id="0",
    product_id="0"
):

    inventory_quantity = max(
        0,
        safe_number(
            inventory_quantity
        )
    )

    order_quantity = max(
        0,
        safe_number(
            order_quantity
        )
    )

    daily_sales = max(
        0,
        safe_number(
            daily_sales
        )
    )

    incoming_stock = max(
        0,
        safe_number(
            incoming_stock
        )
    )

    lead_time = max(
        0,
        safe_number(
            lead_time
        )
    )

    vehicle_capacity = max(
        0,
        safe_number(
            vehicle_capacity
        )
    )

    now = datetime.now()

    def id_number(value):
        match = re.search(r"\d+", str(value))
        return float(match.group()) if match else 0.0

    return {
        "warehouse_code": id_number(warehouse_id),
        "product_code": id_number(product_id),
        "inventory_quantity":
            inventory_quantity,

        "daily_sales":
            daily_sales,

        "incoming_stock":
            incoming_stock,

        # Cold-start defaults: the current daily sales value is used when
        # a live system has not yet supplied product-level sales history.
        "sales_lag_1": safe_number(daily_sales),
        "sales_lag_7": safe_number(daily_sales),
        "sales_lag_14": safe_number(daily_sales),
        "sales_mean_7": safe_number(daily_sales),
        "sales_mean_30": safe_number(daily_sales),
        "day_of_week": now.weekday(),
        "month": now.month,
        "is_weekend": int(now.weekday() >= 5)
    }


# =====================================
# FALLBACK
# =====================================

def fallback(
    daily_sales,
    order_quantity,
    incoming_stock
):

    daily_sales = safe_number(
        daily_sales
    )

    order_quantity = safe_number(
        order_quantity
    )

    incoming_stock = safe_number(
        incoming_stock
    )

    value = (
        daily_sales * 0.6 +
        order_quantity * 0.3 +
        incoming_stock * 0.1
    )

    return max(
        0,
        int(round(value))
    )


# =====================================
# MAIN PREDICT FUNCTION
# =====================================

def predict_demand(**kwargs):

    try:

        features = build_features(
            inventory_quantity=kwargs.get(
                "inventory_quantity",
                0
            ),

            order_quantity=kwargs.get(
                "order_quantity",
                0
            ),

            daily_sales=kwargs.get(
                "daily_sales",
                0
            ),

            incoming_stock=kwargs.get(
                "incoming_stock",
                0
            ),

            lead_time=kwargs.get(
                "lead_time",
                0
            ),

            delivery_status=kwargs.get(
                "delivery_status",
                "Pending"
            ),

            vehicle_capacity=kwargs.get(
                "vehicle_capacity",
                0
            ),

            warehouse_id=kwargs.get(
                "warehouse_id",
                0
            ),

            product_id=kwargs.get(
                "product_id",
                0
            ),
        )

        # =========================
        # NO MODEL
        # =========================

        if model is None:

            return fallback(
                kwargs.get(
                    "daily_sales",
                    0
                ),

                kwargs.get(
                    "order_quantity",
                    0
                ),

                kwargs.get(
                    "incoming_stock",
                    0
                )
            )

        # =========================
        # MODEL PREDICT
        # =========================

        df = pd.DataFrame(
            [features]
        )

        df = df.reindex(
            columns=FEATURE_COLUMNS,
            fill_value=0
        )

        df = df.astype(
            float
        )

        prediction = model.predict(
            df
        )

        if prediction is None:

            return 0

        value = float(
            prediction[0]
        )

        value = int(
            round(value)
        )

        return max(
            0,
            min(
                value,
                100000
            )
        )

    except Exception as e:

        print(
            "[AI ERROR]",
            e
        )

        return fallback(
            kwargs.get(
                "daily_sales",
                0
            ),

            kwargs.get(
                "order_quantity",
                0
            ),

            kwargs.get(
                "incoming_stock",
                0
            )
        )
