# Toronto Collision Risk Prediction & Safe Routing

COMP-4475 Topics in Artificial Intelligence — Accident Prediction Project

Uses the City of Toronto's KSI (Killed or Seriously Injured) dataset (2006–2023) to predict collision fatality risk and find safer driving routes using A\* pathfinding.

## How It Works

1. **Machine Learning Pipeline**: Trains 3 robust, hyperparameter-tuned classifiers (Logistic Regression, Random Forest, Gradient Boosting) on ~18k deduplicated accident records to predict collision risk points.
2. **Real-World Graph Integration**: Dynamically downloads and caches the real driving road network for Toronto using **OSMnx** and constructs a graph network mapped back to risk scores.
3. **Multi-Objective A\* Pathfinding**: Leverages heuristic-driven search algorithms customized for distance, travel time (fastest), and mathematical risk limits (safest) to compute multiple travel routes between any given starting and ending locations.
4. **Modern Analytical Dashboard**: Renders a dynamic, dark-mode GUI using `customtkinter` with live map embedding to interactively geo-locate custom real-world addresses and directly visualize the visual trade-off metrics (time vs distance vs accident risk reduction).
5. **Detailed Visualizations**: Generates dynamic diagnostics, including Precision-Recall curves, feature importances, heatmaps, and spatial analyses.

## Project Structure

```text
config.py              Settings, hyperparameter grid bounds, shared config
main.py                Runs the full ML data pipeline and saves outputs
interactive.py         Modern GUI dashboard for real-world address routing
data/KSI.csv           Toronto KSI collision dataset
src/data.py            Data loading and feature engineering
src/models.py          Training, cross-validation, PR-curves, threshold tuning
src/plots.py           Suite of diagnostic and exploratory visualization plots
src/routing.py         Multi-Objective graph reduction logic and OSMnx processing
outputs/               Generated plots, network cache, and model artifacts
```

## Setup

Requires Python 3.9+ and the KSI dataset at `data/KSI.csv`.

```bash
pip install -r requirements.txt
python main.py           # run the full pipeline
python interactive.py    # launch the interactive GUI (needs main.py output)
```

## Results Summary

- The Random Forest and Gradient Boosting ensembles are capable of strong predictive accuracy across the balanced dataset.
- Real-time routing computations mathematically demonstrate substantial percentage reductions in risk when pivoting to the **Safest** routes against slight or moderate compromises to actual travel distance and ETA compared to the **Fastest** route.

## Technologies

**Core**: Python, pandas, NumPy, scikit-learn  
**Geospatial & Mapping**: OSMnx, NetworkX, Contextily  
**Visualization UI**: matplotlib, seaborn, CustomTkinter (TkAgg embedded canvas)
