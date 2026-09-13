import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.image_preprocessing import create_image_dataloaders, CLASSES, IDX_TO_CLASS
from src.models.train_cnn_model import DefectCNN

FEATURE_COLS = [
    'Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
    'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s',
    'Temp_Diff_K', 'Thermal_Stress_Index', 'Power_kW', 
    'Tool_Wear_Speed_Ratio', 'Vibration_Pressure_Product', 'Torque_Wear_Product'
]

def evaluate_ml_models():
    save_dir = 'saved_models'
    rf_path = os.path.join(save_dir, 'rf_machine_failure.joblib')
    gb_path = os.path.join(save_dir, 'gb_machine_failure.joblib')
    scaler_path = os.path.join(save_dir, 'feature_scaler.joblib')
    
    if not (os.path.exists(rf_path) and os.path.exists(gb_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Trained ML models or scaler missing. Run src/models/train_ml_model.py first.")
        
    rf_model = joblib.load(rf_path)
    gb_model = joblib.load(gb_path)
    scaler = joblib.load(scaler_path)
    
    # Load test data
    data_path = os.path.join('data', 'processed', 'sensor_features_full.csv')
    df = pd.read_csv(data_path)
    
    from sklearn.model_selection import train_test_split
    X = df[FEATURE_COLS]
    y = df['Machine_Failure']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Probabilities & High-Recall Threshold (0.35)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_preds = (rf_probs >= 0.35).astype(int)
    
    gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_preds = (gb_probs >= 0.35).astype(int)
    
    results = {
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, rf_preds)),
            'precision': float(precision_score(y_test, rf_preds)),
            'recall': float(recall_score(y_test, rf_preds)),
            'f1_score': float(f1_score(y_test, rf_preds)),
            'roc_auc': float(roc_auc_score(y_test, rf_probs)),
            'confusion_matrix': confusion_matrix(y_test, rf_preds).tolist()
        },
        'gradient_boosting': {
            'accuracy': float(accuracy_score(y_test, gb_preds)),
            'precision': float(precision_score(y_test, gb_preds)),
            'recall': float(recall_score(y_test, gb_preds)),
            'f1_score': float(f1_score(y_test, gb_preds)),
            'roc_auc': float(roc_auc_score(y_test, gb_probs)),
            'confusion_matrix': confusion_matrix(y_test, gb_preds).tolist()
        }
    }
    print("[SUCCESS] ML Models Evaluation Complete.")
    return results

def evaluate_cnn_model():
    model_path = os.path.join('saved_models', 'defect_cnn.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch CNN model missing at {model_path}. Run src/models/train_cnn_model.py first.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DefectCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    _, _, test_loader = create_image_dataloaders(batch_size=32)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    
    results = {
        'overall': {
            'accuracy': float(acc),
            'precision_weighted': float(prec),
            'recall_weighted': float(rec),
            'f1_weighted': float(f1),
            'confusion_matrix': cm
        },
        'per_class_metrics': report_dict
    }
    print("[SUCCESS] CNN Defect Model Evaluation Complete.")
    return results

def run_full_evaluation():
    ml_eval = evaluate_ml_models()
    cnn_eval = evaluate_cnn_model()
    
    full_eval = {
        'tabular_ml_evaluation': ml_eval,
        'visual_cnn_evaluation': cnn_eval
    }
    
    out_dir = 'reports'
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'evaluation_metrics.json')
    
    with open(report_path, 'w') as f:
        json.dump(full_eval, f, indent=2)
        
    print(f"[SUCCESS] Combined Evaluation Metrics saved to {report_path}")

if __name__ == '__main__':
    run_full_evaluation()
