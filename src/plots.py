"""
src/plots.py - All visualization and plotting functions.

Generates 12 plots total:
  01 - Exploratory analysis (6-panel overview)
  02 - Model performance comparison (accuracy, F1, AUC bars)
  03 - ROC curves for all models
  04 - Confusion matrices
  05 - Feature importance (Random Forest top 15)
  07 - Collision risk by hour of day
  08 - Collision risk by day of week
  09 - Risk by road surface condition
  10 - Risk by visibility
  11 - Toronto collision risk heatmap
  12 - Example scenario predictions
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from config import OUTPUT_DIR, PLOT_DPI, MODERATE_THRESHOLD, HIGH_THRESHOLD


def _save(filename):
    """Save current figure and close it."""
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


# ── Exploratory Analysis ──────────────────────────────────


def plot_exploration(df):
    """6-panel EDA: hour, day, road condition, visibility, light, year."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Toronto KSI Collisions - Exploratory Analysis (2006-2023)", fontsize=16
    )

    df["HOUR"].value_counts().sort_index().plot(
        kind="bar", ax=ax[0, 0], color="steelblue"
    )
    ax[0, 0].set(title="By Hour", xlabel="Hour", ylabel="Count")

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df["DATE"].dt.day_name().value_counts().reindex(days).plot(
        kind="bar", ax=ax[0, 1], color="coral"
    )
    ax[0, 1].set(title="By Day of Week")
    ax[0, 1].tick_params(axis="x", rotation=45)

    df["RDSFCOND"].value_counts().head(8).plot(kind="barh", ax=ax[0, 2], color="teal")
    ax[0, 2].set(title="Road Surface Condition")

    df["VISIBILITY"].value_counts().head(8).plot(
        kind="barh", ax=ax[1, 0], color="goldenrod"
    )
    ax[1, 0].set(title="Visibility")

    df["LIGHT"].value_counts().head(8).plot(
        kind="barh", ax=ax[1, 1], color="mediumpurple"
    )
    ax[1, 1].set(title="Light Conditions")

    df["YEAR"].value_counts().sort_index().plot(
        kind="bar", ax=ax[1, 2], color="darkgreen"
    )
    ax[1, 2].set(title="By Year")
    ax[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save("01_exploration.png")
    print("  Saved: 01_exploration.png")


# ── Model Evaluation Plots ────────────────────────────────


def plot_model_comparison(results):
    """Bar charts comparing accuracy, F1, and AUC across models."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (metric, label, color) in enumerate(
        zip(
            ["accuracy", "f1", "auc"],
            ["Accuracy", "F1 Score", "ROC-AUC"],
            ["steelblue", "coral", "teal"],
        )
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
    """ROC curves for all models on the same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves"
    )
    ax.legend(loc="lower right")
    _save("03_roc_curves.png")


def plot_confusion_matrices(results, y_test):
    """Side-by-side confusion matrices for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["Non-Fatal", "Fatal"],
            yticklabels=["Non-Fatal", "Fatal"],
        )
        axes[i].set(title=name, ylabel="Actual", xlabel="Predicted")
    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    _save("04_confusion_matrices.png")


def plot_feature_importance(results, cols):
    """Horizontal bar chart of top 15 features from Random Forest."""
    imp = results["Random Forest"]["model"].feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [cols[i] for i in idx][::-1],
        imp[idx][::-1],
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set(xlabel="Importance", title="Top 15 Features (Random Forest)")
    plt.tight_layout()
    _save("05_feature_importance.png")


def plot_model_evaluation(results, y_test, cols):
    """Generate all 4 model evaluation plots at once."""
    plot_model_comparison(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(results, cols)
    print("  Saved: 02-05 (model evaluation)")


# ── Risk Analysis Plots ───────────────────────────────────


def plot_hourly_risk(df):
    """Dual-axis chart: collision count (bars) + risk score (line) by hour."""
    hr = (
        df.groupby("hour")
        .agg(count=("OBJECTID", "count"), risk=("combined_risk", "mean"))
        .reset_index()
    )

    fig, a1 = plt.subplots(figsize=(12, 6))
    a1.bar(
        hr["hour"], hr["count"], color="steelblue", alpha=0.5, label="Collision count"
    )
    a1.set(xlabel="Hour of Day", ylabel="Collisions")
    a2 = a1.twinx()
    a2.plot(hr["hour"], hr["risk"], "r-o", lw=2.5, label="Risk score")
    a2.set_ylabel("Risk Score", color="red")
    a1.set_title("Collision Risk by Hour of Day")
    a1.legend(loc="upper left")
    a2.legend(loc="upper right")
    _save("07_hourly_risk.png")


def plot_daily_risk(df):
    """Bar chart of average risk by day of week."""
    dr = df.groupby("day_of_week")["combined_risk"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(7), dr.values, color="coral", edgecolor="black")
    ax.set(
        xticks=range(7),
        xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ylabel="Avg Risk Score",
        title="Collision Risk by Day of Week",
    )
    for b, v in zip(bars, dr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    _save("08_daily_risk.png")


def plot_road_condition_risk(df):
    """Horizontal bars: risk by road surface condition."""
    cr = (
        df.groupby("RDSFCOND")
        .agg(risk=("combined_risk", "mean"), n=("OBJECTID", "count"))
        .reset_index()
    )
    cr = cr[cr["n"] >= 50].sort_values("risk")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cr["RDSFCOND"], cr["risk"], color="teal", edgecolor="black")
    ax.set(xlabel="Avg Risk Score", title="Risk by Road Surface Condition")
    plt.tight_layout()
    _save("09_road_condition_risk.png")


def plot_visibility_risk(df):
    """Horizontal bars: risk by visibility condition."""
    vr = (
        df.groupby("VISIBILITY")
        .agg(risk=("combined_risk", "mean"), n=("OBJECTID", "count"))
        .reset_index()
    )
    vr = vr[vr["n"] >= 50].sort_values("risk")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(vr["VISIBILITY"], vr["risk"], color="goldenrod", edgecolor="black")
    ax.set(xlabel="Avg Risk Score", title="Risk by Visibility")
    plt.tight_layout()
    _save("10_visibility_risk.png")


def plot_risk_heatmap(df):
    """Scatter plot of all collisions colored by risk score."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        df["LONGITUDE"],
        df["LATITUDE"],
        c=df["combined_risk"],
        cmap="YlOrRd",
        alpha=0.4,
        s=5,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, label="Risk Score (0=Low, 1=High)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Toronto Collision Risk Map")
    _save("11_risk_heatmap.png")


def plot_risk_analysis(df):
    """Generate all 5 risk analysis plots at once."""
    plot_hourly_risk(df)
    plot_daily_risk(df)
    plot_road_condition_risk(df)
    plot_visibility_risk(df)
    plot_risk_heatmap(df)
    print("  Saved: 07-11 (risk analysis)")


# ── Scenario Predictions Plot ─────────────────────────────


def plot_scenarios(names, risks):
    """Bar chart of example scenario risk predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        (
            "#2ecc71"
            if r < MODERATE_THRESHOLD
            else "#f39c12" if r < HIGH_THRESHOLD else "#e74c3c"
        )
        for r in risks
    ]
    bars = ax.barh(names, risks, color=colors, edgecolor="black")
    ax.set(
        xlabel="Combined Risk Score",
        title="Risk Predictions for Example Scenarios",
        xlim=(0, 1),
    )
    ax.axvline(MODERATE_THRESHOLD, color="orange", ls="--", alpha=0.7, label="Moderate")
    ax.axvline(HIGH_THRESHOLD, color="red", ls="--", alpha=0.7, label="High")
    ax.legend()
    for b, v in zip(bars, risks):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    plt.tight_layout()
    _save("12_scenario_predictions.png")
    print("  Saved: 12_scenario_predictions.png")
