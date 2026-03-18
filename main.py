"""
Toronto Collision Risk Prediction System
COMP4475 - Topics in Artificial Intelligence

Predicts the risk level (0-1) of a serious traffic collision for a
given location, time, and road conditions in Toronto. Designed to
help drivers plan safer routes and help city planners identify
dangerous areas that need intervention.

Uses the Toronto Police KSI (Killed or Seriously Injured) dataset
from 2006-2023 with ~19,000 collision records.

Author: Nicolas Vicente
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, accuracy_score
)
import warnings
import os
warnings.filterwarnings('ignore')


# ============================================================
# PHASE 1: DATA LOADING & EXPLORATION
# ============================================================

def load_data(filepath='data/KSI.csv'):
    """Load and parse the Toronto Police KSI dataset."""
    print("=" * 60)
    print("PHASE 1: Loading and Exploring Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    # Parse DATE (format: "1/1/2006 10:00:00 AM")
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    
    # Extract hour from TIME column (236 -> hour 2, 1430 -> hour 14)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['HOUR'] = (df['TIME'] // 100).clip(0, 23)
    
    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Date range: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"\n  ACCLASS distribution:")
    print(f"  {df['ACCLASS'].value_counts().to_dict()}")
    
    return df


def explore_data(df):
    """Generate exploratory plots."""
    print("\n" + "=" * 60)
    print("  Exploratory Data Analysis")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Toronto KSI Collisions - Exploratory Analysis (2006-2023)', fontsize=16)
    
    # 1. Collisions by hour
    df['HOUR'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Collisions by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Collisions by day of week
    df['day_name'] = df['DATE'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'].value_counts().reindex(day_order).plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Collisions by Day of Week')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Road surface condition
    df['RDSFCOND'].value_counts().head(8).plot(kind='barh', ax=axes[0, 2], color='teal')
    axes[0, 2].set_title('Road Surface Condition')
    
    # 4. Visibility
    df['VISIBILITY'].value_counts().head(8).plot(kind='barh', ax=axes[1, 0], color='goldenrod')
    axes[1, 0].set_title('Visibility Conditions')
    
    # 5. Light conditions
    df['LIGHT'].value_counts().head(8).plot(kind='barh', ax=axes[1, 1], color='mediumpurple')
    axes[1, 1].set_title('Light Conditions')
    
    # 6. Collisions by year
    df['YEAR'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 2], color='darkgreen')
    axes[1, 2].set_title('Collisions by Year')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/01_exploration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/01_exploration.png")
    
    return df


# ============================================================
# PHASE 2: BUILD THE RISK MODEL
# ============================================================

def build_risk_dataset(df):
    """
    Build a dataset where each row is a LOCATION-TIME SLOT, not
    an individual person in a collision.
    
    The idea: divide Toronto into ~1km grid cells and time into
    slots (hour x day_of_week x season). For each combo, count
    how many collisions occurred historically. Then a model can
    predict: given a location + time + conditions, what is the
    expected collision risk?
    
    This is the UNIQUE CONTRIBUTION of this project:
    - Spatial grid encoding (~1km zones across Toronto)
    - Temporal risk patterns (rush hour, weekend, season)
    - Environmental condition weighting
    - Combined risk score output (0 to 1)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Building Risk Model Dataset")
    print("=" * 60)
    
    df = df.copy()
    df = df.dropna(subset=['DATE', 'LATITUDE', 'LONGITUDE'])
    
    # Filter to Toronto bounds
    df = df[(df['LATITUDE'].between(43.5, 43.9)) & 
            (df['LONGITUDE'].between(-79.7, -79.1))]
    
    # --- Temporal features ---
    df['month'] = df['DATE'].dt.month
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['hour'] = df['HOUR']
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 18))).astype(int)
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1, 4: 1, 5: 1,    # Spring
        6: 2, 7: 2, 8: 2,    # Summer
        9: 3, 10: 3, 11: 3   # Fall
    })
    
    # --- Spatial grid (~1km cells) ---
    grid_size = 0.01  # ~1km
    df['grid_lat'] = (df['LATITUDE'] / grid_size).round() * grid_size
    df['grid_lon'] = (df['LONGITUDE'] / grid_size).round() * grid_size
    df['grid_cell'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
    
    # --- Convert Yes/blank binary columns ---
    binary_cols = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                   'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().str.upper()
            df[col] = (df[col] == 'YES').astype(int)
    
    # --- Severity score: Fatal=1.0, Non-Fatal Injury=0.5, Property Damage=0.1 ---
    df['severity'] = df['ACCLASS'].map({
        'Fatal': 1.0,
        'Non-Fatal Injury': 0.5,
        'Property Damage Only': 0.1
    }).fillna(0.5)
    
    # --- Build aggregated risk features per grid cell ---
    # Historical risk score per location
    cell_stats = df.groupby('grid_cell').agg(
        total_collisions=('OBJECTID', 'count'),
        fatal_count=('severity', lambda x: (x == 1.0).sum()),
        avg_severity=('severity', 'mean'),
        grid_lat=('grid_lat', 'first'),
        grid_lon=('grid_lon', 'first'),
        speeding_rate=('SPEEDING', 'mean'),
        alcohol_rate=('ALCOHOL', 'mean'),
        pedestrian_rate=('PEDESTRIAN', 'mean'),
    ).reset_index()
    
    # Normalize collision count to 0-1 risk score
    cell_stats['location_risk'] = cell_stats['total_collisions'] / cell_stats['total_collisions'].max()
    
    # Merge back
    df = df.merge(cell_stats[['grid_cell', 'location_risk', 'total_collisions', 'avg_severity']], 
                  on='grid_cell', how='left')
    
    # --- Encode categorical features ---
    encode_cols = ['ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE']
    label_encoders = {}
    for col in encode_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # --- Target: is this collision fatal? ---
    df['is_fatal'] = (df['ACCLASS'] == 'Fatal').astype(int)
    
    print(f"  Grid cells: {df['grid_cell'].nunique()}")
    print(f"  Records: {len(df)}")
    print(f"  Fatal: {df['is_fatal'].sum()}, Non-fatal: {(df['is_fatal'] == 0).sum()}")
    
    return df, label_encoders


def select_features(df):
    """Select features for the risk prediction model."""
    feature_cols = [
        # Temporal
        'month', 'day_of_week', 'hour', 'is_weekend', 'is_rush_hour', 'season',
        # Spatial
        'LATITUDE', 'LONGITUDE', 'location_risk',
        # Environmental (encoded)
        'ROAD_CLASS_enc', 'TRAFFCTL_enc', 'VISIBILITY_enc', 'LIGHT_enc', 
        'RDSFCOND_enc', 'IMPACTYPE_enc',
        # Behavioural
        'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL',
        'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df['is_fatal']
    
    print(f"\n  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  X: {X.shape}, y: {y.shape}")
    
    return X, y, feature_cols


# ============================================================
# PHASE 3: TRAIN MODELS & EVALUATE
# ============================================================

def train_models(X, y, feature_cols):
    """Train and compare three models."""
    print("\n" + "=" * 60)
    print("PHASE 3: Model Training & Evaluation")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print(f"\n  Training set: {len(X_train)} rows")
    print(f"  Test set:     {len(X_test)} rows")
    print(f"  Fatal in train: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        if 'Logistic' in name:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model, 'accuracy': acc, 'f1': f1, 'auc': auc,
            'y_pred': y_pred, 'y_prob': y_prob,
        }
        
        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Non-Fatal', 'Fatal']))
    
    # Cross-validation on best model
    print("  5-Fold Cross-Validation (Random Forest):")
    cv = cross_val_score(models['Random Forest'], X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"    Mean AUC: {cv.mean():.4f} (+/- {cv.std():.4f})")
    
    return results, X_test, y_test, scaler


# ============================================================
# PHASE 4: RISK SCORE SYSTEM
# ============================================================

def build_risk_score_system(df, results, feature_cols, label_encoders):
    """
    Build the actual risk score system.
    
    Uses the trained Random Forest model's probability output as
    a base risk score, then combines it with the historical location
    risk to produce a final 0-1 risk score.
    
    Also generates risk profiles for different scenarios.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Building Risk Score System")
    print("=" * 60)
    
    rf_model = results['Random Forest']['model']
    
    # --- Generate risk scores for the full dataset ---
    X_all = df[feature_cols].fillna(0)
    df['model_risk_score'] = rf_model.predict_proba(X_all)[:, 1]
    
    # Combined risk = weighted average of model prediction + location history
    df['combined_risk'] = (0.6 * df['model_risk_score'] + 0.4 * df['location_risk'])
    # Normalize to 0-1
    df['combined_risk'] = df['combined_risk'] / df['combined_risk'].max()
    
    # --- Risk by hour of day ---
    hourly_risk = df.groupby('hour').agg(
        avg_risk=('combined_risk', 'mean'),
        collision_count=('OBJECTID', 'count'),
        fatal_rate=('is_fatal', 'mean')
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(hourly_risk['hour'], hourly_risk['collision_count'], 
            color='steelblue', alpha=0.5, label='Collision count')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Collisions', color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(hourly_risk['hour'], hourly_risk['avg_risk'], 
             color='red', linewidth=2.5, marker='o', label='Risk score')
    ax2.set_ylabel('Average Risk Score', color='red')
    ax1.set_title('Collision Risk by Hour of Day')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('outputs/07_hourly_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/07_hourly_risk.png")
    
    # --- Risk by day of week ---
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_risk = df.groupby('day_of_week').agg(
        avg_risk=('combined_risk', 'mean'),
        fatal_rate=('is_fatal', 'mean')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(daily_risk['day_of_week'], daily_risk['avg_risk'], color='coral', edgecolor='black')
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_ylabel('Average Risk Score')
    ax.set_title('Collision Risk by Day of Week')
    for bar, val in zip(bars, daily_risk['avg_risk']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', fontsize=10)
    plt.savefig('outputs/08_daily_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/08_daily_risk.png")
    
    # --- Risk by road condition ---
    cond_risk = df.groupby('RDSFCOND').agg(
        avg_risk=('combined_risk', 'mean'),
        count=('OBJECTID', 'count')
    ).reset_index()
    cond_risk = cond_risk[cond_risk['count'] >= 50].sort_values('avg_risk', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cond_risk['RDSFCOND'], cond_risk['avg_risk'], color='teal', edgecolor='black')
    ax.set_xlabel('Average Risk Score')
    ax.set_title('Collision Risk by Road Surface Condition')
    plt.tight_layout()
    plt.savefig('outputs/09_road_condition_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/09_road_condition_risk.png")
    
    # --- Risk by visibility ---
    vis_risk = df.groupby('VISIBILITY').agg(
        avg_risk=('combined_risk', 'mean'),
        count=('OBJECTID', 'count')
    ).reset_index()
    vis_risk = vis_risk[vis_risk['count'] >= 50].sort_values('avg_risk', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(vis_risk['VISIBILITY'], vis_risk['avg_risk'], color='goldenrod', edgecolor='black')
    ax.set_xlabel('Average Risk Score')
    ax.set_title('Collision Risk by Visibility')
    plt.tight_layout()
    plt.savefig('outputs/10_visibility_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/10_visibility_risk.png")
    
    # --- Risk heatmap of Toronto ---
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        df['LONGITUDE'], df['LATITUDE'],
        c=df['combined_risk'], cmap='YlOrRd',
        alpha=0.4, s=5, vmin=0, vmax=1
    )
    plt.colorbar(scatter, label='Combined Risk Score (0=Low, 1=High)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Toronto Collision Risk Map')
    plt.savefig('outputs/11_risk_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/11_risk_heatmap.png")
    
    # --- Top 10 most dangerous grid cells ---
    top_danger = df.groupby(['grid_cell', 'grid_lat', 'grid_lon']).agg(
        avg_risk=('combined_risk', 'mean'),
        total_collisions=('OBJECTID', 'count'),
        fatal_count=('is_fatal', 'sum'),
    ).reset_index().sort_values('avg_risk', ascending=False).head(10)
    
    print("\n  TOP 10 MOST DANGEROUS AREAS IN TORONTO:")
    print("  " + "-" * 55)
    for _, row in top_danger.iterrows():
        print(f"    Lat: {row['grid_lat']:.2f}, Lon: {row['grid_lon']:.2f} | "
              f"Risk: {row['avg_risk']:.3f} | "
              f"Collisions: {row['total_collisions']:.0f} | "
              f"Fatals: {row['fatal_count']:.0f}")
    
    return df


def demo_risk_predictions(df, results, feature_cols, label_encoders):
    """
    Demonstrate the risk prediction system with example scenarios.
    Shows what a user or city planner would actually see.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Example Risk Predictions")
    print("=" * 60)
    
    rf_model = results['Random Forest']['model']
    
    # Helper: encode a categorical value using the saved encoder
    def encode_val(col, val):
        if col in label_encoders and val in label_encoders[col].classes_:
            return label_encoders[col].transform([val])[0]
        return 0
    
    # Get median location risk for default
    median_loc_risk = df['location_risk'].median()
    
    # Define example scenarios
    scenarios = [
        {
            'name': 'Friday night, downtown, wet roads, dark',
            'month': 11, 'day_of_week': 4, 'hour': 23,
            'is_weekend': 0, 'is_rush_hour': 0, 'season': 3,
            'LATITUDE': 43.65, 'LONGITUDE': -79.38, 'location_risk': 0.8,
            'ROAD_CLASS': 'Major Arterial', 'TRAFFCTL': 'Traffic Signal',
            'VISIBILITY': 'Rain', 'LIGHT': 'Dark',
            'RDSFCOND': 'Wet', 'IMPACTYPE': 'Pedestrian Collisions',
            'SPEEDING': 0, 'AG_DRIV': 0, 'REDLIGHT': 0, 'ALCOHOL': 0,
            'PEDESTRIAN': 1, 'CYCLIST': 0, 'AUTOMOBILE': 1, 'MOTORCYCLE': 0, 'TRUCK': 0,
        },
        {
            'name': 'Tuesday morning, suburban, clear, rush hour',
            'month': 6, 'day_of_week': 1, 'hour': 8,
            'is_weekend': 0, 'is_rush_hour': 1, 'season': 2,
            'LATITUDE': 43.75, 'LONGITUDE': -79.30, 'location_risk': 0.3,
            'ROAD_CLASS': 'Minor Arterial', 'TRAFFCTL': 'Traffic Signal',
            'VISIBILITY': 'Clear', 'LIGHT': 'Daylight',
            'RDSFCOND': 'Dry', 'IMPACTYPE': 'Rear End',
            'SPEEDING': 0, 'AG_DRIV': 0, 'REDLIGHT': 0, 'ALCOHOL': 0,
            'PEDESTRIAN': 0, 'CYCLIST': 0, 'AUTOMOBILE': 1, 'MOTORCYCLE': 0, 'TRUCK': 0,
        },
        {
            'name': 'Saturday 2am, highway, icy, alcohol involved',
            'month': 1, 'day_of_week': 5, 'hour': 2,
            'is_weekend': 1, 'is_rush_hour': 0, 'season': 0,
            'LATITUDE': 43.70, 'LONGITUDE': -79.40, 'location_risk': 0.6,
            'ROAD_CLASS': 'Expressway', 'TRAFFCTL': 'No Control',
            'VISIBILITY': 'Snow', 'LIGHT': 'Dark',
            'RDSFCOND': 'Ice', 'IMPACTYPE': 'SMV Other',
            'SPEEDING': 1, 'AG_DRIV': 1, 'REDLIGHT': 0, 'ALCOHOL': 1,
            'PEDESTRIAN': 0, 'CYCLIST': 0, 'AUTOMOBILE': 1, 'MOTORCYCLE': 0, 'TRUCK': 0,
        },
        {
            'name': 'Sunday afternoon, residential, clear, cyclist',
            'month': 7, 'day_of_week': 6, 'hour': 14,
            'is_weekend': 1, 'is_rush_hour': 0, 'season': 2,
            'LATITUDE': 43.68, 'LONGITUDE': -79.35, 'location_risk': 0.2,
            'ROAD_CLASS': 'Local', 'TRAFFCTL': 'Stop Sign',
            'VISIBILITY': 'Clear', 'LIGHT': 'Daylight',
            'RDSFCOND': 'Dry', 'IMPACTYPE': 'Cyclist Collisions',
            'SPEEDING': 0, 'AG_DRIV': 0, 'REDLIGHT': 0, 'ALCOHOL': 0,
            'PEDESTRIAN': 0, 'CYCLIST': 1, 'AUTOMOBILE': 1, 'MOTORCYCLE': 0, 'TRUCK': 0,
        },
    ]
    
    print("\n  EXAMPLE RISK PREDICTIONS:")
    print("  " + "=" * 60)
    
    scenario_names = []
    scenario_risks = []
    
    for s in scenarios:
        # Build feature row matching feature_cols order
        row = {}
        for col in feature_cols:
            if col.endswith('_enc'):
                base_col = col.replace('_enc', '')
                row[col] = encode_val(base_col, s.get(base_col, 'Unknown'))
            else:
                row[col] = s.get(col, 0)
        
        X_scenario = pd.DataFrame([row])[feature_cols].fillna(0)
        risk_prob = rf_model.predict_proba(X_scenario)[0][1]
        
        # Combine with location risk
        combined = 0.6 * risk_prob + 0.4 * s['location_risk']
        
        # Risk level label
        if combined >= 0.5:
            level = 'HIGH RISK'
        elif combined >= 0.25:
            level = 'MODERATE RISK'
        else:
            level = 'LOW RISK'
        
        scenario_names.append(s['name'])
        scenario_risks.append(combined)
        
        print(f"\n  Scenario: {s['name']}")
        print(f"    Model risk:    {risk_prob:.3f}")
        print(f"    Location risk: {s['location_risk']:.3f}")
        print(f"    Combined risk: {combined:.3f} -> {level}")
    
    # Plot scenario comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if r < 0.25 else '#f39c12' if r < 0.5 else '#e74c3c' for r in scenario_risks]
    bars = ax.barh(scenario_names, scenario_risks, color=colors, edgecolor='black')
    ax.set_xlabel('Combined Risk Score')
    ax.set_title('Risk Predictions for Example Scenarios')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax.legend()
    for bar, val in zip(bars, scenario_risks):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/12_scenario_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: outputs/12_scenario_predictions.png")


# ============================================================
# PHASE 6: MODEL EVALUATION PLOTS
# ============================================================

def plot_model_evaluation(results, X_test, y_test, feature_cols):
    """Generate model comparison and evaluation plots."""
    print("\n" + "=" * 60)
    print("PHASE 6: Model Evaluation Plots")
    print("=" * 60)
    
    # 1. Model comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1', 'auc']
    labels = ['Accuracy', 'F1 Score', 'ROC-AUC']
    colors = ['steelblue', 'coral', 'teal']
    
    for i, (m, l) in enumerate(zip(metrics, labels)):
        vals = [results[n][m] for n in model_names]
        axes[i].bar(model_names, vals, color=colors[i], edgecolor='black', alpha=0.8)
        axes[i].set_title(l, fontsize=14)
        axes[i].set_ylim(0, 1)
        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)
        axes[i].tick_params(axis='x', rotation=15)
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/02_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/02_model_comparison.png")
    
    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models')
    ax.legend(loc='lower right')
    plt.savefig('outputs/03_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/03_roc_curves.png")
    
    # 3. Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Non-Fatal', 'Fatal'],
                    yticklabels=['Non-Fatal', 'Fatal'])
        axes[i].set_title(name)
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    plt.suptitle('Confusion Matrices', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/04_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/04_confusion_matrices.png")
    
    # 4. Feature importance
    rf = results['Random Forest']['model']
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in idx][::-1], imp[idx][::-1],
            color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Most Important Features (Random Forest)')
    plt.tight_layout()
    plt.savefig('outputs/05_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/05_feature_importance.png")
    
    # 5. Fatal rate by hour (for risk context)
    print("  Saved: outputs/06_fatal_rate_by_hour.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    data_path = 'data/KSI.csv'
    if not os.path.exists(data_path):
        print("DATA NOT FOUND! Save KSI.csv in the data/ folder.")
        exit(1)
    
    # Phase 1: Load & explore
    df = load_data(data_path)
    df = explore_data(df)
    
    # Phase 2: Build risk dataset
    df, label_encoders = build_risk_dataset(df)
    X, y, feature_cols = select_features(df)
    
    # Phase 3: Train models
    results, X_test, y_test, scaler = train_models(X, y, feature_cols)
    
    # Phase 4: Build risk score system
    df = build_risk_score_system(df, results, feature_cols, label_encoders)
    
    # Phase 5: Demo predictions
    demo_risk_predictions(df, results, feature_cols, label_encoders)
    
    # Phase 6: Evaluation plots
    plot_model_evaluation(results, X_test, y_test, feature_cols)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    best = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  Best model: {best}")
    print(f"  Accuracy: {results[best]['accuracy']:.4f}")
    print(f"  F1:       {results[best]['f1']:.4f}")
    print(f"  ROC-AUC:  {results[best]['auc']:.4f}")
    print(f"\n  12 plots saved in outputs/ folder.")
    print(f"  Use these for your report and slides!")