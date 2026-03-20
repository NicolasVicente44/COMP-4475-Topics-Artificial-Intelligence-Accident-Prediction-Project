"""
main.py - Entry point for the Toronto Collision Risk Prediction System.

Run with:  python main.py

This script ties together the data, models, and plots modules to:
  1. Load and explore the KSI dataset
  2. Engineer temporal, spatial, and environmental features
  3. Train and compare 3 ML models (Logistic Regression, Random Forest, Gradient Boosting)
  4. Compute combined risk scores for every collision
  5. Demo the system with example scenarios
  6. Generate 12 publication-ready plots in the outputs/ folder
"""

import os
from config import DATA_PATH, OUTPUT_DIR
from src.data import load_data, engineer_features
from src.models import train_models, compute_risk_scores, get_dangerous_areas, predict_scenario
from src.plots import plot_exploration, plot_model_evaluation, plot_risk_analysis, plot_scenarios


# ── Example scenarios for demonstrating the risk system ────
# Each scenario represents a realistic driving situation a user
# or city planner might want to evaluate.

SCENARIOS = [
    (
        "Friday night, downtown, wet, dark",
        dict(
            month=11, day_of_week=4, hour=23, is_weekend=0, is_rush_hour=0, season=3,
            LATITUDE=43.65, LONGITUDE=-79.38, location_risk=0.8,
            ROAD_CLASS="Major Arterial", TRAFFCTL="Traffic Signal",
            VISIBILITY="Rain", LIGHT="Dark", RDSFCOND="Wet",
            IMPACTYPE="Pedestrian Collisions",
            SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
            PEDESTRIAN=1, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
        ),
    ),
    (
        "Tuesday morning, suburban, clear",
        dict(
            month=6, day_of_week=1, hour=8, is_weekend=0, is_rush_hour=1, season=2,
            LATITUDE=43.75, LONGITUDE=-79.30, location_risk=0.3,
            ROAD_CLASS="Minor Arterial", TRAFFCTL="Traffic Signal",
            VISIBILITY="Clear", LIGHT="Daylight", RDSFCOND="Dry",
            IMPACTYPE="Rear End",
            SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
            PEDESTRIAN=0, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
        ),
    ),
    (
        "Saturday 2am, highway, icy, alcohol",
        dict(
            month=1, day_of_week=5, hour=2, is_weekend=1, is_rush_hour=0, season=0,
            LATITUDE=43.70, LONGITUDE=-79.40, location_risk=0.6,
            ROAD_CLASS="Expressway", TRAFFCTL="No Control",
            VISIBILITY="Snow", LIGHT="Dark", RDSFCOND="Ice",
            IMPACTYPE="SMV Other",
            SPEEDING=1, AG_DRIV=1, REDLIGHT=0, ALCOHOL=1,
            PEDESTRIAN=0, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
        ),
    ),
    (
        "Sunday afternoon, residential, cyclist",
        dict(
            month=7, day_of_week=6, hour=14, is_weekend=1, is_rush_hour=0, season=2,
            LATITUDE=43.68, LONGITUDE=-79.35, location_risk=0.2,
            ROAD_CLASS="Local", TRAFFCTL="Stop Sign",
            VISIBILITY="Clear", LIGHT="Daylight", RDSFCOND="Dry",
            IMPACTYPE="Cyclist Collisions",
            SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
            PEDESTRIAN=0, CYCLIST=1, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
        ),
    ),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Place KSI.csv in {DATA_PATH}")
        return

    print("=" * 60)
    print("  Toronto Collision Risk Prediction System")
    print("=" * 60)

    # ── Step 1: Load & explore ─────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data(DATA_PATH)
    plot_exploration(df)

    # ── Step 2: Feature engineering ────────────────────────
    print("\n[2/5] Engineering features...")
    df, encoders, cols = engineer_features(df)

    # ── Step 3: Train models ───────────────────────────────
    print("\n[3/5] Training models...")
    results, X_test, y_test = train_models(df, cols)

    # ── Step 4: Risk scores & demo ─────────────────────────
    print("\n[4/5] Computing risk scores...")
    df = compute_risk_scores(df, results, cols)

    # Print most dangerous areas
    top = get_dangerous_areas(df)
    print("\n  TOP 10 MOST DANGEROUS AREAS:")
    for _, r in top.iterrows():
        print(f"    ({r['grid_lat']:.2f}, {r['grid_lon']:.2f}) | "
              f"Risk: {r['risk']:.3f} | Collisions: {r['collisions']} | Fatals: {r['fatals']}")

    # Run example scenario predictions
    print("\n  Example predictions:")
    rf = results["Random Forest"]["model"]
    names, risks = [], []
    for name, scenario in SCENARIOS:
        prob, combined, level = predict_scenario(rf, cols, encoders, scenario)
        names.append(name)
        risks.append(combined)
        print(f"    {name}: {combined:.3f} -> {level} RISK")
    plot_scenarios(names, risks)

    # ── Step 5: Generate all plots ─────────────────────────
    print("\n[5/5] Generating plots...")
    plot_risk_analysis(df)
    plot_model_evaluation(results, y_test, cols)

    # ── Summary ────────────────────────────────────────────
    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n{'=' * 60}")
    print(f"  DONE! Best model: {best}")
    print(f"  Accuracy: {results[best]['accuracy']:.4f}")
    print(f"  F1: {results[best]['f1']:.4f} | AUC: {results[best]['auc']:.4f}")
    print(f"  12 plots saved in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
