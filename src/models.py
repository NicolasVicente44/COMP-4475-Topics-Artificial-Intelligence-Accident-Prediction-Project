"""
src/models.py - Model training, evaluation, and risk scoring.

Handles:
  - Training Logistic Regression, Random Forest, Gradient Boosting
  - Evaluating with accuracy, F1, ROC-AUC, and classification reports
  - Cross-validation for generalization check
  - Computing combined risk scores (model prediction + location history)
  - Running example scenario predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from config import (
    TEST_SIZE,
    RANDOM_STATE,
    RF_TREES,
    RF_DEPTH,
    GB_TREES,
    GB_DEPTH,
    GB_LR,
    MODEL_WEIGHT,
    LOCATION_WEIGHT,
    MODERATE_THRESHOLD,
    HIGH_THRESHOLD,
)


def train_models(df, cols):
    """
    Train three classifiers and compare performance.

    Uses stratified split to handle the class imbalance (only ~14%
    of collisions are fatal). Logistic Regression gets scaled features;
    tree-based models use raw values.

    Returns:
        results: Dict with model objects and metrics per model
        X_test:  Test features (for plotting later)
        y_test:  Test labels  (for plotting later)
    """
    X = df[cols].fillna(0)
    y = df["is_fatal"]

    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale for Logistic Regression (tree models don't need it)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # Define models (name -> (model, needs_scaling))
    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=RF_TREES,
                max_depth=RF_DEPTH,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=GB_TREES,
                max_depth=GB_DEPTH,
                learning_rate=GB_LR,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }

    # Train and evaluate each model
    results = {}
    for name, (model, use_scaled) in models.items():
        Xf = X_tr_sc if use_scaled else X_tr
        Xt = X_te_sc if use_scaled else X_te

        model.fit(Xf, y_tr)
        y_pred = model.predict(Xt)
        y_prob = model.predict_proba(Xt)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="weighted")
        auc = roc_auc_score(y_te, y_prob)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        print(f"\n  {name}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
        print(classification_report(y_te, y_pred, target_names=["Non-Fatal", "Fatal"]))

    # Cross-validation on Random Forest
    cv = cross_val_score(
        models["Random Forest"][0], X, y, cv=5, scoring="roc_auc", n_jobs=-1
    )
    print(f"  Random Forest 5-Fold CV AUC: {cv.mean():.4f} (+/- {cv.std():.4f})")

    return results, X_te, y_te


def compute_risk_scores(df, results, cols):
    """
    Add a combined risk score (0-1) to every row in the dataset.

    Combined risk = weighted blend of:
      - Model prediction (probability of fatality from Random Forest)
      - Location risk (historical collision density for that grid cell)

    Weights are set in config.py (default: 60% model, 40% location).
    """
    rf = results["Random Forest"]["model"]
    df["model_risk"] = rf.predict_proba(df[cols].fillna(0))[:, 1]
    df["combined_risk"] = (
        MODEL_WEIGHT * df["model_risk"] + LOCATION_WEIGHT * df["location_risk"]
    )
    df["combined_risk"] = df["combined_risk"] / df["combined_risk"].max()
    return df


def get_dangerous_areas(df, top_n=10):
    """Return the top N most dangerous grid cells by average risk."""
    return (
        df.groupby(["grid_cell", "grid_lat", "grid_lon"])
        .agg(
            risk=("combined_risk", "mean"),
            collisions=("OBJECTID", "count"),
            fatals=("is_fatal", "sum"),
        )
        .reset_index()
        .sort_values("risk", ascending=False)
        .head(top_n)
    )


def predict_scenario(rf_model, cols, encoders, scenario):
    """
    Predict risk for a single scenario dict.

    The scenario dict should contain values for each feature column.
    Categorical features (ending in _enc) are encoded using the
    saved LabelEncoders from training.

    Returns (model_risk, combined_risk, risk_level).
    """

    def enc(col, val):
        if col in encoders and val in encoders[col].classes_:
            return encoders[col].transform([val])[0]
        return 0

    # Build feature row
    row = {}
    for c in cols:
        if c.endswith("_enc"):
            base = c.replace("_enc", "")
            row[c] = enc(base, scenario.get(base, "Unknown"))
        else:
            row[c] = scenario.get(c, 0)

    X = pd.DataFrame([row])[cols].fillna(0)
    prob = rf_model.predict_proba(X)[0][1]
    combined = MODEL_WEIGHT * prob + LOCATION_WEIGHT * scenario.get("location_risk", 0)

    if combined >= HIGH_THRESHOLD:
        level = "HIGH"
    elif combined >= MODERATE_THRESHOLD:
        level = "MODERATE"
    else:
        level = "LOW"

    return prob, combined, level
