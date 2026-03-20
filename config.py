"""
config.py - All project settings in one place.

Change these values to adjust the model behavior, grid resolution,
risk thresholds, or which features to use. No need to touch other files.
"""

# ── Paths ──────────────────────────────────────────────────
DATA_PATH = "data/KSI.csv"
OUTPUT_DIR = "outputs"

# ── Toronto geographic bounds (filter out bad coordinates) ─
LAT_MIN, LAT_MAX = 43.5, 43.9
LON_MIN, LON_MAX = -79.7, -79.1

# ── Spatial grid resolution ────────────────────────────────
# 0.01 degrees ≈ 1km cells. Smaller = more precise but fewer
# collisions per cell. Larger = smoother but less granular.
GRID_SIZE = 0.01

# ── Risk score weights ─────────────────────────────────────
# How much the model prediction vs historical location risk
# contributes to the final combined score. Must sum to 1.0.
MODEL_WEIGHT = 0.6  # Weight for Random Forest prediction
LOCATION_WEIGHT = 0.4  # Weight for historical location risk

# ── Risk thresholds ────────────────────────────────────────
# Used to label predictions as LOW / MODERATE / HIGH
MODERATE_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.50

# ── Model parameters ──────────────────────────────────────
TEST_SIZE = 0.2  # Fraction of data used for testing
RANDOM_STATE = 42  # Seed for reproducibility
RF_TREES = 200  # Number of trees in Random Forest
RF_DEPTH = 15  # Max depth per tree
GB_TREES = 200  # Number of trees in Gradient Boosting
GB_DEPTH = 5  # Max depth per tree
GB_LR = 0.1  # Learning rate for Gradient Boosting

# ── Feature columns ───────────────────────────────────────
# Binary Yes/No columns in the KSI dataset (converted to 1/0)
BINARY_COLS = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
]

# Categorical columns to label-encode for the model
ENCODE_COLS = [
    "ROAD_CLASS",
    "TRAFFCTL",
    "VISIBILITY",
    "LIGHT",
    "RDSFCOND",
    "IMPACTYPE",
]

# All features the model uses (temporal + spatial + encoded + binary)
FEATURE_COLS = (
    [
        "month",
        "day_of_week",
        "hour",
        "is_weekend",
        "is_rush_hour",
        "season",
        "LATITUDE",
        "LONGITUDE",
        "location_risk",
    ]
    + [c + "_enc" for c in ENCODE_COLS]
    + BINARY_COLS
)

# ── Plot settings ─────────────────────────────────────────
PLOT_DPI = 150
