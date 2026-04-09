# src/data.py - Data loading, cleaning, and feature engineering.

import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import (
    OUTPUT_DIR,
    BINARY_COLS,
    ENCODE_COLS,
    FEATURE_COLS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    GRID_SIZE,
)


def load_data(path):
    """Load the KSI CSV and parse date/time fields."""
    if not os.path.exists(path):
        print(f"Error: Data file not found at '{path}'.")
        print("Please ensure the dataset exists before running the script.")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["YEAR"] = df["DATE"].dt.year

    # TIME column is an int like 236 (2:36 AM) or 1430 (2:30 PM)
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
    df["HOUR"] = (df["TIME"] // 100).clip(0, 23)

    print(f"  Loaded {len(df)} records ({df['YEAR'].min()}-{df['YEAR'].max()})")
    print(f"  Classes: {df['ACCLASS'].value_counts().to_dict()}")
    return df


def engineer_features(df):
    """Clean data and build all features for the models."""
    df = df.dropna(subset=["DATE", "LATITUDE", "LONGITUDE"]).copy()
    df = df[
        df["LATITUDE"].between(LAT_MIN, LAT_MAX)
        & df["LONGITUDE"].between(LON_MIN, LON_MAX)
    ]

    # Temporal features
    df["month"] = df["DATE"].dt.month
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["hour"] = df["HOUR"]
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].between(7, 9) | df["hour"].between(16, 18)).astype(int)
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1, 4: 1, 5: 1,    # Spring
        6: 2, 7: 2, 8: 2,    # Summer
        9: 3, 10: 3, 11: 3,  # Fall
    })

    # Spatial grid (~1km cells), normalize collision count to 0-1
    df["grid_lat"] = (df["LATITUDE"] / GRID_SIZE).round() * GRID_SIZE
    df["grid_lon"] = (df["LONGITUDE"] / GRID_SIZE).round() * GRID_SIZE
    df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    counts = df.groupby("grid_cell").size().reset_index(name="cell_count")
    df = df.merge(counts, on="grid_cell", how="left")
    df["location_risk"] = df["cell_count"] / df["cell_count"].max()

    # Binary columns (Yes/blank -> 1/0)
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = (
                df[c].fillna("").astype(str).str.strip().str.upper() == "YES"
            ).astype(int)

    # Encode categoricals
    encoders = {}
    for c in ENCODE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c])
            encoders[c] = le

    df["is_fatal"] = (df["ACCLASS"] == "Fatal").astype(int)

    # Deduplication one row per accident for training.
    # max() for binary cols so flags are preserved if any person had them.
    print(f"  Before dedup: {len(df)} rows | {df['ACCNUM'].nunique()} unique accidents")

    agg_dict = {c: "first" for c in df.columns if c != "ACCNUM"}
    for c in BINARY_COLS:
        if c in df.columns:
            agg_dict[c] = "max"
    agg_dict["is_fatal"] = "max"

    df_dedup = df.groupby("ACCNUM").agg(agg_dict).reset_index()

    print(f"  After dedup:  {len(df_dedup)} rows (one per accident)")

    cols = [c for c in FEATURE_COLS if c in df.columns]

    print(f"  {df['grid_cell'].nunique()} grid cells | {len(cols)} features")
    print(f"  Fatal: {df_dedup['is_fatal'].sum()} | Non-fatal: {(df_dedup['is_fatal'] == 0).sum()}")

    return df, df_dedup, encoders, cols


def save_risk_grid(df):
    """Save the risk grid CSV for the routing module."""
    ROUTE_GRID = 0.005  # ~500m cells

    gdf = df[["LATITUDE", "LONGITUDE", "combined_risk", "OBJECTID", "is_fatal"]].copy()
    gdf["glat"] = (gdf["LATITUDE"] / ROUTE_GRID).round() * ROUTE_GRID
    gdf["glon"] = (gdf["LONGITUDE"] / ROUTE_GRID).round() * ROUTE_GRID

    grid_data = (
        gdf.groupby(["glat", "glon"])
        .agg(risk=("combined_risk", "mean"), count=("OBJECTID", "count"),
             fatals=("is_fatal", "sum"), fatal_ratio=("is_fatal", "mean"))
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