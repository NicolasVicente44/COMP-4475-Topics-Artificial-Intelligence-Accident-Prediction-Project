# Toronto Collision Risk Prediction & Safe Routing

COMP-4475 Topics in Artificial Intelligence — Accident Prediction Project

Uses the City of Toronto's KSI (Killed or Seriously Injured) dataset (2006–2023) to predict collision fatality risk and find safer driving routes using A* pathfinding.

## How It Works

1. Trains 3 classifiers (Logistic Regression, Random Forest, Gradient Boosting) on ~18k deduplicated accident records
2. Builds a spatial risk grid (~1km cells) with per-cell collision density
3. Combines ML prediction (60%) with historical location risk (40%) into a final score
4. Runs A* search to find both shortest and safest paths between Toronto landmarks
5. Generates 14 plots covering EDA, model evaluation, risk analysis, and routing

## Project Structure

```
config.py              Settings, hyperparameters, and demo data
main.py                Runs the full pipeline
interactive.py         Tkinter GUI for exploring safe routes
data/KSI.csv           Toronto KSI collision dataset
src/data.py            Data loading and feature engineering
src/models.py          Training, evaluation, risk scoring
src/plots.py           All 14 visualization functions
src/routing.py         A* pathfinding on the risk grid
outputs/               Generated plots and risk grid CSV
```

## Setup

Requires Python 3.9+ and the KSI dataset at `data/KSI.csv`.

```bash
pip install -r requirements.txt
python main.py           # run the full pipeline
python interactive.py    # launch the GUI (needs main.py output)
```

## Results

- Best model: Random Forest (Acc ~86.9%, F1 ~0.823, AUC ~0.681)
- Safe routing reduces risk by ~50% with ~38% extra distance on average

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Tkinter
