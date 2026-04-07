# src/plots.py - All visualization and plotting functions.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from config import OUTPUT_DIR, PLOT_DPI, MODERATE_THRESHOLD, HIGH_THRESHOLD, SHORELINE_LONS, SHORELINE_LATS


def _save(filename):
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_exploration(df):
    """6-panel EDA overview."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Toronto KSI Collisions - Exploratory Analysis (2006-2023)", fontsize=16)

    df["HOUR"].value_counts().sort_index().plot(kind="bar", ax=ax[0, 0], color="steelblue")
    ax[0, 0].set(title="By Hour", xlabel="Hour", ylabel="Count")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["DATE"].dt.day_name().value_counts().reindex(days).plot(kind="bar", ax=ax[0, 1], color="coral")
    ax[0, 1].set(title="By Day of Week")
    ax[0, 1].tick_params(axis="x", rotation=45)

    df["RDSFCOND"].value_counts().head(8).plot(kind="barh", ax=ax[0, 2], color="teal")
    ax[0, 2].set(title="Road Surface Condition")

    df["VISIBILITY"].value_counts().head(8).plot(kind="barh", ax=ax[1, 0], color="goldenrod")
    ax[1, 0].set(title="Visibility")

    df["LIGHT"].value_counts().head(8).plot(kind="barh", ax=ax[1, 1], color="mediumpurple")
    ax[1, 1].set(title="Light Conditions")

    df["YEAR"].value_counts().sort_index().plot(kind="bar", ax=ax[1, 2], color="darkgreen")
    ax[1, 2].set(title="By Year")
    ax[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save("01_exploration.png")
    print("  Saved: 01_exploration.png")


def plot_model_comparison(results):
    """Bar charts comparing accuracy, F1, and AUC across models."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (metric, label, color) in enumerate(
        zip(["accuracy", "f1", "auc"], ["Accuracy", "F1 Score", "ROC-AUC"],
            ["steelblue", "coral", "teal"])
    ):
        vals = [results[n][metric] for n in names]
        axes[i].bar(names, vals, color=color, edgecolor="black", alpha=0.8)
        axes[i].set(title=label, ylim=(0, 1))
        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center")
        axes[i].tick_params(axis="x", rotation=15)

    plt.suptitle("Model Performance Comparison", fontsize=16)
    plt.tight_layout()
    _save("02_model_comparison.png")


def plot_roc_curves(results, y_test):
    """ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right")
    _save("03_roc_curves.png")


def plot_confusion_matrices(results, y_test):
    """Side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Non-Fatal", "Fatal"], yticklabels=["Non-Fatal", "Fatal"])
        axes[i].set(title=name, ylabel="Actual", xlabel="Predicted")
    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    _save("04_confusion_matrices.png")


def plot_feature_importance(results, cols):
    """Top 15 feature importances from Random Forest."""
    imp = results["Random Forest"]["model"].feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([cols[i] for i in idx][::-1], imp[idx][::-1],
            color="steelblue", edgecolor="black", alpha=0.8)
    ax.set(xlabel="Importance", title="Top 15 Features (Random Forest)")
    plt.tight_layout()
    _save("05_feature_importance.png")


def plot_model_evaluation(results, y_test, cols):
    """Generate all 4 model evaluation plots."""
    plot_model_comparison(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(results, cols)
    print("  Saved: 02-05 (model evaluation)")


def plot_hourly_risk(df):
    """Collision count + risk score by hour (dual axis)."""
    hr = df.groupby("hour").agg(
        count=("OBJECTID", "count"), risk=("combined_risk", "mean")
    ).reset_index()

    fig, a1 = plt.subplots(figsize=(12, 6))
    a1.bar(hr["hour"], hr["count"], color="steelblue", alpha=0.5, label="Collision count")
    a1.set(xlabel="Hour of Day", ylabel="Collisions")
    a2 = a1.twinx()
    a2.plot(hr["hour"], hr["risk"], "r-o", lw=2.5, label="Risk score")
    a2.set_ylabel("Risk Score", color="red")
    a1.set_title("Collision Risk by Hour of Day")
    a1.legend(loc="upper left")
    a2.legend(loc="upper right")
    _save("06_hourly_risk.png")


def plot_daily_risk(df):
    """Average risk by day of week."""
    dr = df.groupby("day_of_week")["combined_risk"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(7), dr.values, color="coral", edgecolor="black")
    ax.set(xticks=range(7), xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
           ylabel="Avg Risk Score", title="Collision Risk by Day of Week")
    for b, v in zip(bars, dr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    _save("07_daily_risk.png")


def plot_road_condition_risk(df):
    """Risk by road surface condition."""
    _plot_categorical_risk(df, "RDSFCOND", "teal", "08_road_condition_risk.png",
                           "Risk by Road Surface Condition")


def plot_visibility_risk(df):
    """Risk by visibility condition."""
    _plot_categorical_risk(df, "VISIBILITY", "goldenrod", "09_visibility_risk.png",
                           "Risk by Visibility")


def _plot_categorical_risk(df, col, color, filename, title):
    data = df.groupby(col).agg(
        risk=("combined_risk", "mean"), n=("OBJECTID", "count")
    ).reset_index()
    data = data[data["n"] >= 50].sort_values("risk")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data[col], data["risk"], color=color, edgecolor="black")
    ax.set(xlabel="Avg Risk Score", title=title)
    plt.tight_layout()
    _save(filename)


def plot_risk_heatmap(df):
    """Toronto-wide collision risk scatter map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(df["LONGITUDE"], df["LATITUDE"], c=df["combined_risk"],
                    cmap="YlOrRd", alpha=0.4, s=5, vmin=0, vmax=1)
    plt.colorbar(sc, label="Risk Score (0=Low, 1=High)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Toronto Collision Risk Map")
    _save("10_risk_heatmap.png")


def plot_risk_analysis(df):
    """Generate all 5 risk analysis plots."""
    plot_hourly_risk(df)
    plot_daily_risk(df)
    plot_road_condition_risk(df)
    plot_visibility_risk(df)
    plot_risk_heatmap(df)
    print("  Saved: 06-10 (risk analysis)")


def plot_scenarios(names, risks):
    """Bar chart of scenario risk predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#2ecc71" if r < MODERATE_THRESHOLD
        else "#f39c12" if r < HIGH_THRESHOLD else "#e74c3c"
        for r in risks
    ]
    bars = ax.barh(names, risks, color=colors, edgecolor="black")
    ax.set(xlabel="Combined Risk Score", title="Risk Predictions for Example Scenarios", xlim=(0, 1))
    ax.axvline(MODERATE_THRESHOLD, color="orange", ls="--", alpha=0.7, label="Moderate")
    ax.axvline(HIGH_THRESHOLD, color="red", ls="--", alpha=0.7, label="High")
    ax.legend()
    for b, v in zip(bars, risks):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    plt.tight_layout()
    _save("11_scenario_predictions.png")
    print("  Saved: 11_scenario_predictions.png")


def plot_route_comparison(all_routes):
    """Side-by-side distance and risk comparison for shortest vs safest routes."""
    n = len(all_routes)
    names = [r["name"] for r in all_routes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = range(n)
    w = 0.35

    # Distance
    d_short = [r["shortest"]["distance_km"] for r in all_routes]
    d_safe = [r["safest"]["distance_km"] for r in all_routes]
    ax1.bar([i - w/2 for i in x], d_short, w, color="#ef4444", alpha=0.8, label="Shortest", edgecolor="black")
    ax1.bar([i + w/2 for i in x], d_safe, w, color="#22c55e", alpha=0.8, label="Safest", edgecolor="black")
    ax1.set(xticks=list(x), xticklabels=names, ylabel="Distance (km)", title="Route Distance Comparison")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend()
    for i in x:
        ax1.text(i - w/2, d_short[i] + 0.3, f"{d_short[i]:.1f}", ha="center", fontsize=8)
        ax1.text(i + w/2, d_safe[i] + 0.3, f"{d_safe[i]:.1f}", ha="center", fontsize=8)

    # Risk
    r_short = [r["shortest"]["risk_sum"] for r in all_routes]
    r_safe = [r["safest"]["risk_sum"] for r in all_routes]
    ax2.bar([i - w/2 for i in x], r_short, w, color="#ef4444", alpha=0.8, label="Shortest", edgecolor="black")
    ax2.bar([i + w/2 for i in x], r_safe, w, color="#22c55e", alpha=0.8, label="Safest", edgecolor="black")
    ax2.set(xticks=list(x), xticklabels=names, ylabel="Cumulative Risk Score", title="Route Risk Comparison")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend()
    for i in x:
        ax2.text(i - w/2, r_short[i] + 0.1, f"{r_short[i]:.2f}", ha="center", fontsize=8)
        ax2.text(i + w/2, r_safe[i] + 0.1, f"{r_safe[i]:.2f}", ha="center", fontsize=8)

    plt.suptitle("A* Pathfinding: Shortest vs Safest Routes", fontsize=16)
    plt.tight_layout()
    _save("12_route_comparison.png")
    print("  Saved: 12_route_comparison.png")


def plot_route_map(all_routes, df):
    """Route panels showing shortest vs safest paths on collision heatmap."""
    n = len(all_routes)
    cols = 3
    rows = (n + cols) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(22, 14))
    axes = axes.flatten()

    for i, route in enumerate(all_routes):
        ax = axes[i]

        ax.scatter(df["LONGITUDE"], df["LATITUDE"], c=df["combined_risk"],
                   cmap="YlOrRd", alpha=0.12, s=2, vmin=0, vmax=1)

        # Lake Ontario
        ax.fill_between(SHORELINE_LONS, SHORELINE_LATS, 43.55, color="#3B82F6", alpha=0.08)
        ax.plot(SHORELINE_LONS, SHORELINE_LATS, color="#3B82F6", linewidth=1, alpha=0.3)

        # Shortest (red dashed)
        short_path = route["shortest"]["path"]
        slats = [p[0] for p in short_path]
        slons = [p[1] for p in short_path]
        ax.plot(slons, slats, color="#e74c3c", linewidth=2.5, linestyle="--",
                alpha=0.9, label=f"Shortest: {route['shortest']['distance_km']:.1f}km", zorder=3)

        # Safest (green solid)
        safe_path = route["safest"]["path"]
        flats = [p[0] for p in safe_path]
        flons = [p[1] for p in safe_path]
        ax.plot(flons, flats, color="#2ecc71", linewidth=3,
                alpha=0.95, label=f"Safest: {route['safest']['distance_km']:.1f}km", zorder=4)

        # Markers
        ax.scatter(slons[0], slats[0], c="#3498db", s=100, zorder=5,
                   edgecolors="white", linewidths=2, marker="o")
        ax.annotate("START", (slons[0], slats[0]), fontsize=7, fontweight="bold",
                    color="#3498db", xytext=(5, 8), textcoords="offset points")
        ax.scatter(slons[-1], slats[-1], c="#e67e22", s=120, zorder=5,
                   edgecolors="white", linewidths=2, marker="*")
        ax.annotate("END", (slons[-1], slats[-1]), fontsize=7, fontweight="bold",
                    color="#e67e22", xytext=(5, 8), textcoords="offset points")

        # Zoom to route area
        all_lats = slats + flats
        all_lons = slons + flons
        pad_lat = (max(all_lats) - min(all_lats)) * 0.25 + 0.01
        pad_lon = (max(all_lons) - min(all_lons)) * 0.25 + 0.01
        ax.set_xlim(min(all_lons) - pad_lon, max(all_lons) + pad_lon)
        ax.set_ylim(min(all_lats) - pad_lat, max(all_lats) + pad_lat)

        ax.set_title(
            f"{route['name']}\n"
            f"Risk: -{route['risk_reduction']*100:.0f}%  |  "
            f"Distance: +{route['distance_increase']*100:.0f}%",
            fontsize=11, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots, add legend in last cell
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[n].text(
        0.5, 0.5,
        "Legend\n\n"
        "--- Red dashed = Shortest path\n"
        "___ Green solid = Safest path\n\n"
        "Blue dot = Start\n"
        "Orange star = End\n\n"
        "Background: collision heatmap\n"
        "(darker = more dangerous)",
        transform=axes[n].transAxes, ha="center", va="center",
        fontsize=12, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.suptitle("Toronto Safe Routing: A* Pathfinding Results",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("13_route_map.png")
    print("  Saved: 13_route_map.png")


def plot_risk_reduction_summary(all_routes):
    """Risk reduction % vs extra distance % for each route."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r["name"] for r in all_routes]
    reductions = [r["risk_reduction"] * 100 for r in all_routes]
    extras = [r["distance_increase"] * 100 for r in all_routes]

    x = range(len(names))
    w = 0.35

    bars1 = ax.bar([i - w/2 for i in x], reductions, w,
                   color="#22c55e", alpha=0.85, label="Risk Reduction (%)", edgecolor="black")
    bars2 = ax.bar([i + w/2 for i in x], extras, w,
                   color="#f59e0b", alpha=0.85, label="Extra Distance (%)", edgecolor="black")

    for b, v in zip(bars1, reductions):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold", color="#22c55e")
    for b, v in zip(bars2, extras):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold", color="#f59e0b")

    avg_r = sum(reductions) / len(reductions)
    avg_e = sum(extras) / len(extras)
    ax.axhline(avg_r, color="#22c55e", linestyle=":", alpha=0.5)
    ax.axhline(avg_e, color="#f59e0b", linestyle=":", alpha=0.5)
    ax.text(len(names) - 0.5, avg_r + 1, f"Avg: {avg_r:.1f}%", color="#22c55e", fontsize=8)
    ax.text(len(names) - 0.5, avg_e + 1, f"Avg: {avg_e:.1f}%", color="#f59e0b", fontsize=8)

    ax.set(xticks=list(x), xticklabels=names, ylabel="Percentage (%)",
           title="Safe Routing Trade-off: Risk Reduction vs Extra Distance")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    _save("14_risk_reduction_summary.png")
    print("  Saved: 14_risk_reduction_summary.png")