"""
main.py - Entry point for the Toronto Collision Risk Prediction & Safe Routing System.

Run with:  python main.py

This script ties together the data, models, routing, and plots modules to:
  1. Load and explore the KSI dataset
  2. Engineer temporal, spatial, and environmental features
  3. Train and compare 3 ML models (Logistic Regression, Random Forest, Gradient Boosting)
  4. Compute combined risk scores for every collision
  5. Build a spatial risk grid and run A* pathfinding for safe routing
  6. Demo the system with example scenarios
  7. Generate publication-ready plots in the outputs/ folder
"""

import os
from config import DATA_PATH, OUTPUT_DIR
from src.data import load_data, engineer_features
from src.models import train_models, compute_risk_scores, get_dangerous_areas, predict_scenario
from src.plots import (
    plot_exploration,
    plot_model_evaluation,
    plot_risk_analysis,
    plot_scenarios,
    plot_route_comparison,
    plot_route_map,
    plot_risk_reduction_summary,
)
from src.routing import RiskGrid, compare_routes


# ── Example scenarios for demonstrating the risk system ────

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


# ── Example routes for demonstrating A* safe routing ───────

DEMO_ROUTES = [
    ("Downtown to Scarborough",     (43.6550, -79.3830), (43.7730, -79.2580)),
    ("Etobicoke to East York",      (43.6440, -79.5100), (43.6920, -79.3270)),
    ("North York to Waterfront",    (43.7670, -79.4110), (43.6390, -79.3810)),
    ("Yorkdale to Beaches",         (43.7250, -79.4520), (43.6670, -79.2930)),
    ("Pearson Area to Downtown",    (43.6800, -79.6100), (43.6500, -79.3800)),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Place KSI.csv in {DATA_PATH}")
        return

    print("=" * 60)
    print("  Toronto Collision Risk Prediction & Safe Routing System")
    print("=" * 60)

    # ── Step 1: Load & explore ─────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_data(DATA_PATH)
    plot_exploration(df)

    # ── Step 2: Feature engineering ────────────────────────
    print("\n[2/6] Engineering features...")
    df, encoders, cols = engineer_features(df)

    # ── Step 3: Train models ───────────────────────────────
    print("\n[3/6] Training models...")
    results, X_test, y_test = train_models(df, cols)

    # ── Step 4: Risk scores & scenarios ────────────────────
    print("\n[4/6] Computing risk scores...")
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

    # ── Step 5: A* Safe Routing ────────────────────────────
    print("\n[5/6] Building risk grid & running A* pathfinding...")

    # Save the risk grid for the routing module
    _save_risk_grid(df)

    # Build the routing graph
    grid = RiskGrid(f"{OUTPUT_DIR}/risk_grid.csv")

    # Run all demo routes
    all_routes = []
    for route_name, start, goal in DEMO_ROUTES:
        print(f"\n  Route: {route_name}")
        result = compare_routes(grid, start, goal)
        if result:
            result["name"] = route_name
            all_routes.append(result)

    # Generate routing plots
    if all_routes:
        plot_route_comparison(all_routes)
        plot_route_map(all_routes, df)
        plot_risk_reduction_summary(all_routes)

    # ── Step 6: Generate all remaining plots ───────────────
    print("\n[6/6] Generating plots...")
    plot_risk_analysis(df)
    plot_model_evaluation(results, y_test, cols)

    # ── Summary ────────────────────────────────────────────
    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n{'=' * 60}")
    print(f"  DONE! Best model: {best}")
    print(f"  Accuracy: {results[best]['accuracy']:.4f}")
    print(f"  F1: {results[best]['f1']:.4f} | AUC: {results[best]['auc']:.4f}")
    if all_routes:
        avg_reduction = sum(r["risk_reduction"] for r in all_routes) / len(all_routes)
        avg_extra = sum(r["distance_increase"] for r in all_routes) / len(all_routes)
        print(f"  Avg risk reduction: {avg_reduction*100:.1f}% | Avg extra distance: {avg_extra*100:.1f}%")
    print(f"  Plots saved in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


def _save_risk_grid(df):
    """Save the risk grid CSV needed by the routing module."""
    ROUTE_GRID = 0.005  # Finer grid for routing (~500m)

    df["glat"] = (df["LATITUDE"] / ROUTE_GRID).round() * ROUTE_GRID
    df["glon"] = (df["LONGITUDE"] / ROUTE_GRID).round() * ROUTE_GRID

    grid_data = (
        df.groupby(["glat", "glon"])
        .agg(
            risk=("combined_risk", "mean"),
            count=("OBJECTID", "count"),
            fatals=("is_fatal", "sum"),
            fatal_ratio=("is_fatal", "mean"),
        )
        .reset_index()
    )

    # Composite route risk: model risk + fatality rate + volume
    grid_data["route_risk"] = (
        0.5 * grid_data["risk"]
        + 0.3 * grid_data["fatal_ratio"]
        + 0.2 * (grid_data["count"] / grid_data["count"].max())
    )
    grid_data["route_risk"] = grid_data["route_risk"] / grid_data["route_risk"].max()

    grid_data.to_csv(f"{OUTPUT_DIR}/risk_grid.csv", index=False)
    print(f"  Risk grid saved: {len(grid_data)} cells")


if __name__ == "__main__":
    main()