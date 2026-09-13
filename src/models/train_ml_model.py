import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import add_telematic_features

FEATURE_COLS = [
    'Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
    'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s',
    'Temp_Diff_K', 'Thermal_Stress_Index', 'Power_kW', 
    'Tool_Wear_Speed_Ratio', 'Vibration_Pressure_Product', 'Torque_Wear_Product'
]
TARGET_COL = 'Machine_Failure'

def train_ml_models(data_path=os.path.join('data', 'processed', 'sensor_features_full.csv'), random_seed=42):
    if not os.path.exists(data_path):
        print(f"[INFO] Engineered dataset not found at {data_path}. Generating on the fly...")
        from src.features.feature_engineering import process_and_save_features
        df = process_and_save_features()
    else:
        df = pd.read_csv(data_path)
        
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=random_seed, stratify=y_train_val
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, 'feature_scaler.joblib'))
    
    print("\n" + "="*50)
    print("TRAINING TABULAR MACHINE FAILURE CLASSIFIERS")
    print("="*50)
    
    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        class_weight='balanced',
        random_state=random_seed
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_probs_test = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # High-recall threshold tuning (e.g. threshold 0.35 to catch high-risk failures early)
    rf_preds_opt = (rf_probs_test >= 0.35).astype(int)
    
    print("\n--- RANDOM FOREST RESULTS (Decision Threshold = 0.35) ---")
    print(f"Accuracy : {accuracy_score(y_test, rf_preds_opt):.4f}")
    print(f"Precision: {precision_score(y_test, rf_preds_opt):.4f}")
    print(f"Recall   : {recall_score(y_test, rf_preds_opt):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, rf_probs_test):.4f}")
    
    # 2. Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=random_seed
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_probs_test = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_preds_opt = (gb_probs_test >= 0.35).astype(int)
    
    print("\n--- GRADIENT BOOSTING RESULTS (Decision Threshold = 0.35) ---")
    print(f"Accuracy : {accuracy_score(y_test, gb_preds_opt):.4f}")
    print(f"Precision: {precision_score(y_test, gb_preds_opt):.4f}")
    print(f"Recall   : {recall_score(y_test, gb_preds_opt):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, gb_probs_test):.4f}")
    
    # Save Best Models
    joblib.dump(rf_model, os.path.join(save_dir, 'rf_machine_failure.joblib'))
    joblib.dump(gb_model, os.path.join(save_dir, 'gb_machine_failure.joblib'))
    
    print(f"\n[SUCCESS] Trained ML models saved to {save_dir}/")
    return rf_model, gb_model, scaler

if __name__ == '__main__':
    train_ml_models()
