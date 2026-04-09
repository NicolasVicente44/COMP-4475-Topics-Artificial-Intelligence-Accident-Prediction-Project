# main.py - Runs the full pipeline.
# Usage: python main.py

import os
from config import DATA_PATH, OUTPUT_DIR, SCENARIOS, DEMO_ROUTES
from src.data import load_data, engineer_features, save_risk_grid
from src.models import (
    train_models,
    compute_risk_scores,
    get_dangerous_areas,
    predict_scenario,
)
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


# Main entry point to run the full pipeline for the accident prediction and safe routing system
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if the data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Place KSI.csv in {DATA_PATH}")
        return

    # Print the title of the application
    print("=" * 60)
    print("  Toronto Collision Risk Prediction & Safe Routing System")
    print("=" * 60)

    # Load the data
    print("\n[1/6] Loading data...")
    df = load_data(DATA_PATH)
    plot_exploration(df)

    # Engineer features
    print("\n[2/6] Engineering features...")
    df, df_dedup, encoders, cols = engineer_features(df)

    # Train models
    print("\n[3/6] Training models...")
    results, X_test, y_test = train_models(df_dedup, cols)

    # Compute risk scores
    print("\n[4/6] Computing risk scores...")
    df = compute_risk_scores(df, results, cols)

    # Get the top 10 most dangerous areas
    top = get_dangerous_areas(df)
    print("\n  TOP 10 MOST DANGEROUS AREAS:")
    for _, r in top.iterrows():
        print(
            f"    ({r['grid_lat']:.2f}, {r['grid_lon']:.2f}) | "
            f"Risk: {r['risk']:.3f} | Collisions: {r['collisions']} | Fatals: {r['fatals']}"
        )

    # Predict scenarios
    print("\n  Example predictions:")
    rf = results["Random Forest"]["model"]
    names, risks = [], []
    for name, scenario in SCENARIOS:
        prob, combined, level = predict_scenario(rf, cols, encoders, scenario)
        names.append(name)
        risks.append(combined)
        print(f"    {name}: {combined:.3f} -> {level} RISK")
    plot_scenarios(names, risks)

    # Build risk grid & run A* pathfinding
    print("\n[5/6] Building risk grid & running A* pathfinding...")
    save_risk_grid(df)
    grid = RiskGrid(f"{OUTPUT_DIR}/risk_grid.csv")

    # Compare routes
    all_routes = []
    for route_name, start, goal in DEMO_ROUTES:
        print(f"\n  Route: {route_name}")
        result = compare_routes(grid, start, goal)
        if result:
            result["name"] = route_name
            all_routes.append(result)

    # Plot route comparison
    if all_routes:
        plot_route_comparison(all_routes)
        plot_route_map(all_routes, df)
        plot_risk_reduction_summary(all_routes)

    # Generate plots
    print("\n[6/6] Generating plots...")
    plot_risk_analysis(df)
    plot_model_evaluation(results, y_test, cols)

    # Print the best model
    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n{'=' * 60}")
    print(f"  Best model: {best}")
    print(f"  Accuracy: {results[best]['accuracy']:.4f}")
    print(f"  F1: {results[best]['f1']:.4f} | AUC: {results[best]['auc']:.4f}")
    if all_routes:
        avg_reduction = sum(r["risk_reduction"] for r in all_routes) / len(all_routes)
        avg_extra = sum(r["distance_increase"] for r in all_routes) / len(all_routes)
        print(
            f"  Avg risk reduction: {avg_reduction*100:.1f}% | Avg extra distance: {avg_extra*100:.1f}%"
        )
    print(f"  Plots saved in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


# Main entry point to run the full pipeline for the accident prediction and safe routing grid creation system
if __name__ == "__main__":
    main()
