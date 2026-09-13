import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.feature_engineering import add_telematic_features
from src.preprocessing.image_preprocessing import get_image_transforms, CLASSES, IDX_TO_CLASS
from src.models.train_cnn_model import DefectCNN

FEATURE_COLS = [
    'Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
    'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s',
    'Temp_Diff_K', 'Thermal_Stress_Index', 'Power_kW', 
    'Tool_Wear_Speed_Ratio', 'Vibration_Pressure_Product', 'Torque_Wear_Product'
]

class ManufacturingInferenceEngine:
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ML model & scaler
        rf_path = os.path.join(model_dir, 'rf_machine_failure.joblib')
        scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
        
        if not (os.path.exists(rf_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError("ML model or scaler missing. Run training pipeline first.")
            
        self.ml_model = joblib.load(rf_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load PyTorch CNN
        cnn_path = os.path.join(model_dir, 'defect_cnn.pth')
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"PyTorch CNN missing at {cnn_path}.")
            
        self.cnn_model = DefectCNN(num_classes=len(CLASSES)).to(self.device)
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn_model.eval()
        
        _, self.eval_transform = get_image_transforms()
        print("[INFO] Manufacturing Inference Engine initialized successfully.")

    def predict_machine_status(self, raw_sensor_dict):
        """
        Input: dict with raw sensor keys:
        Air_Temperature_K, Process_Temperature_K, Rotational_Speed_RPM, Torque_Nm, Tool_Wear_min, Pressure_bar, Vibration_mm_s
        """
        df_raw = pd.DataFrame([raw_sensor_dict])
        df_feat = add_telematic_features(df_raw)
        X = df_feat[FEATURE_COLS]
        X_scaled = self.scaler.transform(X)
        
        prob = float(self.ml_model.predict_proba(X_scaled)[0, 1])
        # High-recall decision boundary at 0.35
        is_failure = prob >= 0.35
        
        # Heuristic primary failure mode identification
        temp_diff = raw_sensor_dict.get('Process_Temperature_K', 300) - raw_sensor_dict.get('Air_Temperature_K', 290)
        tool_wear = raw_sensor_dict.get('Tool_Wear_min', 0)
        vibration = raw_sensor_dict.get('Vibration_mm_s', 0)
        torque = raw_sensor_dict.get('Torque_Nm', 0)
        
        if is_failure:
            if tool_wear >= 190:
                failure_mode = 'Tool Wear Accumulation'
            elif temp_diff < 8.8:
                failure_mode = 'Heat Dissipation Failure'
            elif vibration > 9.5 or torque > 65:
                failure_mode = 'Overstrain & Mechanical Stress'
            else:
                failure_mode = 'Power System Failure'
        else:
            failure_mode = 'Normal Operation'
            
        return {
            'machine_status': 'Failure Risk' if is_failure else 'Normal Operation',
            'failure_probability_pct': round(prob * 100, 1),
            'raw_probability': prob,
            'primary_failure_mode': failure_mode
        }

    def predict_product_defect(self, image_input):
        """
        Input: PIL Image object or file path string
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        img_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.cnn_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            
        pred_class = IDX_TO_CLASS[pred_idx]
        is_defective = pred_class != 'normal'
        
        all_class_probs = {IDX_TO_CLASS[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASSES))}
        
        return {
            'product_status': 'Defective' if is_defective else 'Normal Product',
            'defect_type': pred_class.replace('_', ' ').title(),
            'defect_confidence_pct': round(conf * 100, 1),
            'raw_confidence': conf,
            'class_probabilities': all_class_probs
        }

    def generate_joint_report(self, raw_sensor_dict, image_input):
        machine_res = self.predict_machine_status(raw_sensor_dict)
        product_res = self.predict_product_defect(image_input)
        
        machine_fail = machine_res['machine_status'] == 'Failure Risk'
        product_def = product_res['product_status'] == 'Defective'
        
        if machine_fail and product_def:
            overall_status = 'CRITICAL ALERT'
            action = 'QUARANTINE PRODUCT BATCH & HALT MACHINE FOR IMMEDIATE THERMAL & TOOL MAINTENANCE'
        elif machine_fail:
            overall_status = 'MACHINE WARNING'
            action = 'SCHEDULE PREDICTIVE MAINTENANCE (HIGH BREAKDOWN RISK)'
        elif product_def:
            overall_status = 'QUALITY ALERT'
            action = 'REJECT & QUARANTINE DEFECTIVE PRODUCT BATCH'
        else:
            overall_status = 'SYSTEM NORMAL'
            action = 'PASS INSPECTION & CONTINUE STANDARD PRODUCTION'
            
        report = {
            'overall_status': overall_status,
            'machine_health': machine_res,
            'product_quality': product_res,
            'recommended_action': action
        }
        return report

if __name__ == '__main__':
    engine = ManufacturingInferenceEngine()
    
    # Test Sample Telemetry
    sample_telemetry = {
        'Air_Temperature_K': 298.2,
        'Process_Temperature_K': 308.5,
        'Rotational_Speed_RPM': 1420.0,
        'Torque_Nm': 68.5,
        'Tool_Wear_min': 210,
        'Pressure_bar': 7.2,
        'Vibration_mm_s': 11.4
    }
    
    # Test Sample Image
    sample_img_path = os.path.join('data', 'raw', 'images', 'crack', 'crack_001.png')
    
    if os.path.exists(sample_img_path):
        report = engine.generate_joint_report(sample_telemetry, sample_img_path)
        print("\n" + "="*60)
        print("SAMPLE JOINT MANUFACTURING INFERENCE REPORT")
        print("="*60)
        print(f"Overall Status   : {report['overall_status']}")
        print(f"Machine Status   : {report['machine_health']['machine_status']}")
        print(f"Failure Prob     : {report['machine_health']['failure_probability_pct']}%")
        print(f"Failure Mode     : {report['machine_health']['primary_failure_mode']}")
        print(f"Product Status   : {report['product_quality']['product_status']}")
        print(f"Defect Type      : {report['product_quality']['defect_type']}")
        print(f"Defect Confidence: {report['product_quality']['defect_confidence_pct']}%")
        print(f"Action           : {report['recommended_action']}")
        print("="*60)
