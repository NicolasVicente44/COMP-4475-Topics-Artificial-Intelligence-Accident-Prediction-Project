"""
src/data.py - Data loading, cleaning, and feature engineering.

Handles:
  - Loading and parsing the raw KSI CSV
  - Cleaning bad/missing values
  - Creating temporal features (hour, day, month, season, rush hour)
  - Creating spatial grid features (~1km cells with risk scores)
  - Encoding categorical and binary columns
  - Defining the target variable (fatal vs non-fatal)
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import (
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
    """
    Load the KSI CSV and parse date/time fields.

    The TIME column is an integer like 236 (meaning 2:36 AM)
    or 1430 (meaning 2:30 PM). We extract the hour by integer
    dividing by 100.
    """
    if not os.path.exists(path):
        print(f"Error: Data file not found at '{path}'.")
        print("Please ensure the dataset exists before running the script.")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

    # Parse date string ("1/1/2006 10:00:00 AM" -> datetime)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["YEAR"] = df["DATE"].dt.year

    # Extract hour from TIME (236 -> 2, 1430 -> 14)
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
    df["HOUR"] = (df["TIME"] // 100).clip(0, 23)

    print(f"  Loaded {len(df)} records ({df['YEAR'].min()}-{df['YEAR'].max()})")
    print(f"  Classes: {df['ACCLASS'].value_counts().to_dict()}")
    return df


def engineer_features(df):
    """
    Clean the data and create all features used by the models.

    Returns:
        df:       Cleaned DataFrame with new feature columns
        encoders: Dict of LabelEncoder objects (needed for predictions)
        cols:     List of feature column names actually used
    """
    # ── Drop bad rows ──────────────────────────────────────
    df = df.dropna(subset=["DATE", "LATITUDE", "LONGITUDE"]).copy()
    df = df[
        df["LATITUDE"].between(LAT_MIN, LAT_MAX)
        & df["LONGITUDE"].between(LON_MIN, LON_MAX)
    ]

    # ── Temporal features ──────────────────────────────────
    df["month"] = df["DATE"].dt.month
    df["day_of_week"] = df["DATE"].dt.dayofweek  # 0=Mon, 6=Sun
    df["hour"] = df["HOUR"]
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].between(7, 9) | df["hour"].between(16, 18)).astype(
        int
    )
    df["season"] = df["month"].map(
        {
            12: 0,
            1: 0,
            2: 0,  # Winter
            3: 1,
            4: 1,
            5: 1,  # Spring
            6: 2,
            7: 2,
            8: 2,  # Summer
            9: 3,
            10: 3,
            11: 3,
        }  # Fall
    )

    # ── Spatial grid (~1km cells) ──────────────────────────
    # Round lat/lon to nearest grid point, then count how many
    # collisions fall in each cell. Normalize to 0-1 risk score.
    df["grid_lat"] = (df["LATITUDE"] / GRID_SIZE).round() * GRID_SIZE
    df["grid_lon"] = (df["LONGITUDE"] / GRID_SIZE).round() * GRID_SIZE
    df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    counts = df.groupby("grid_cell").size().reset_index(name="cell_count")
    df = df.merge(counts, on="grid_cell", how="left")
    df["location_risk"] = df["cell_count"] / df["cell_count"].max()

    # ── Binary columns (Yes/blank -> 1/0) ──────────────────
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = (
                df[c].fillna("").astype(str).str.strip().str.upper() == "YES"
            ).astype(int)

    # ── Encode categorical columns ─────────────────────────
    encoders = {}
    for c in ENCODE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c])
            encoders[c] = le

    # ── Target variable ────────────────────────────────────
    df["is_fatal"] = (df["ACCLASS"] == "Fatal").astype(int)

    # ── Deduplicate: one row per accident ──────────────────
    # The raw dataset has multiple rows per accident (one per person
    # involved). Using all rows causes data leakage because rows from
    # the same accident share identical features and labels. We keep
    # the full dataframe (df) for spatial analysis / risk grid, but
    # create a deduplicated version (df_dedup) for model training.
    #
    # For binary columns (SPEEDING, ALCOHOL, etc.), we use max() so
    # that if ANY person in the accident had that flag, it's preserved.
    # For other features, we take the first row since they're identical
    # within an accident.

    print(f"  Before dedup: {len(df)} rows | {df['ACCNUM'].nunique()} unique accidents")

    agg_dict = {c: "first" for c in df.columns if c != "ACCNUM"}
    for c in BINARY_COLS:
        if c in df.columns:
            agg_dict[c] = "max"  # preserve if ANY person had the flag
    agg_dict["is_fatal"] = "max"  # fatal if ANY person died

    df_dedup = df.groupby("ACCNUM").agg(agg_dict).reset_index()

    print(f"  After dedup:  {len(df_dedup)} rows (one per accident)")

    # Only keep features that actually exist in the dataframe
    cols = [c for c in FEATURE_COLS if c in df.columns]

    print(
        f"  {df['grid_cell'].nunique()} grid cells | {len(cols)} features"
    )
    print(f"  Fatal: {df_dedup['is_fatal'].sum()} | Non-fatal: {(df_dedup['is_fatal'] == 0).sum()}")

    return df, df_dedup, encoders, cols