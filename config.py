# config.py - Project settings and hyperparameters.

# Paths
DATA_PATH = "data/KSI.csv"
OUTPUT_DIR = "outputs"

# Toronto geographic bounds
LAT_MIN, LAT_MAX = 43.5, 43.9
LON_MIN, LON_MAX = -79.7, -79.1

# Lake Ontario shoreline (piecewise-linear: lon -> min land latitude)
SHORELINE_LONS = [-79.65, -79.55, -79.50, -79.45, -79.40, -79.38,
                  -79.35, -79.30, -79.25, -79.20, -79.15, -79.10]
SHORELINE_LATS = [43.58, 43.585, 43.60, 43.62, 43.63, 43.635,
                  43.64, 43.66, 43.69, 43.73, 43.74, 43.75]

# Grid resolution (1km cells)
GRID_SIZE = 0.01

# Risk score weights (sum to 1.0)
MODEL_WEIGHT = 0.6
LOCATION_WEIGHT = 0.4

# Risk thresholds for LOW / MODERATE / HIGH
MODERATE_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.50

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_TREES = 200
RF_DEPTH = 15
GB_TREES = 200
GB_DEPTH = 5
GB_LR = 0.1
XGB_TREES = 200
XGB_DEPTH = 5
XGB_LR = 0.1

# Binary Yes/No columns in the KSI dataset
BINARY_COLS = [
    "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
    "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL",
]

# Categorical columns to label-encode
ENCODE_COLS = [
    "ROAD_CLASS", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND",
    "IMPACTYPE", "DRIVACT", "DRIVCOND", "MANOEUVER", "DISTRICT",
]

# All features used by the model
FEATURE_COLS = (
    ["month", "day_of_week", "hour", "is_weekend", "is_rush_hour",
     "season", "LATITUDE", "LONGITUDE", "location_risk"]
    + [c + "_enc" for c in ENCODE_COLS]
    + BINARY_COLS
)

PLOT_DPI = 150

# Example scenarios for risk prediction demo
SCENARIOS = [
    ("Friday night, downtown, wet, dark", dict(
        month=11, day_of_week=4, hour=23, is_weekend=0, is_rush_hour=0, season=3,
        LATITUDE=43.65, LONGITUDE=-79.38, location_risk=0.8,
        ROAD_CLASS="Major Arterial", TRAFFCTL="Traffic Signal",
        VISIBILITY="Rain", LIGHT="Dark", RDSFCOND="Wet",
        IMPACTYPE="Pedestrian Collisions",
        SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
        PEDESTRIAN=1, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
    )),
    ("Tuesday morning, suburban, clear", dict(
        month=6, day_of_week=1, hour=8, is_weekend=0, is_rush_hour=1, season=2,
        LATITUDE=43.75, LONGITUDE=-79.30, location_risk=0.3,
        ROAD_CLASS="Minor Arterial", TRAFFCTL="Traffic Signal",
        VISIBILITY="Clear", LIGHT="Daylight", RDSFCOND="Dry",
        IMPACTYPE="Rear End",
        SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
        PEDESTRIAN=0, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
    )),
    ("Saturday 2am, highway, icy, alcohol", dict(
        month=1, day_of_week=5, hour=2, is_weekend=1, is_rush_hour=0, season=0,
        LATITUDE=43.70, LONGITUDE=-79.40, location_risk=0.6,
        ROAD_CLASS="Expressway", TRAFFCTL="No Control",
        VISIBILITY="Snow", LIGHT="Dark", RDSFCOND="Ice",
        IMPACTYPE="SMV Other",
        SPEEDING=1, AG_DRIV=1, REDLIGHT=0, ALCOHOL=1,
        PEDESTRIAN=0, CYCLIST=0, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
    )),
    ("Sunday afternoon, residential, cyclist", dict(
        month=7, day_of_week=6, hour=14, is_weekend=1, is_rush_hour=0, season=2,
        LATITUDE=43.68, LONGITUDE=-79.35, location_risk=0.2,
        ROAD_CLASS="Local", TRAFFCTL="Stop Sign",
        VISIBILITY="Clear", LIGHT="Daylight", RDSFCOND="Dry",
        IMPACTYPE="Cyclist Collisions",
        SPEEDING=0, AG_DRIV=0, REDLIGHT=0, ALCOHOL=0,
        PEDESTRIAN=0, CYCLIST=1, AUTOMOBILE=1, MOTORCYCLE=0, TRUCK=0,
    )),
]

# Demo routes for A* pathfinding
DEMO_ROUTES = [
    ("Downtown to Scarborough",   (43.6550, -79.3830), (43.7730, -79.2580)),
    ("Etobicoke to East York",    (43.6440, -79.5100), (43.6920, -79.3270)),
    ("North York to Waterfront",  (43.7670, -79.4110), (43.6390, -79.3810)),
    ("Yorkdale to Beaches",       (43.7250, -79.4520), (43.6670, -79.2930)),
    ("Pearson Area to Downtown",  (43.6800, -79.6100), (43.6500, -79.3800)),
]
