import os
import numpy as np
import pandas as pd

def add_telematic_features(df):
    """
    Computes domain-specific physical features for machine failure diagnosis.
    """
    df_feat = df.copy()
    
    # 1. Temperature Difference (Heat Dissipation Indicator)
    df_feat['Temp_Diff_K'] = df_feat['Process_Temperature_K'] - df_feat['Air_Temperature_K']
    
    # 2. Thermal Stress Ratio
    df_feat['Thermal_Stress_Index'] = df_feat['Temp_Diff_K'] / (df_feat['Air_Temperature_K'] + 1e-5)
    
    # 3. Mechanical Shaft Power (kW)
    # Power = (2 * pi * RPM * Torque) / 60,000
    df_feat['Power_kW'] = (2 * np.pi * df_feat['Rotational_Speed_RPM'] * df_feat['Torque_Nm']) / 60000.0
    
    # 4. Tool Wear per RPM Unit
    df_feat['Tool_Wear_Speed_Ratio'] = df_feat['Tool_Wear_min'] / (df_feat['Rotational_Speed_RPM'] + 1.0)
    
    # 5. Vibration & Pressure Interaction
    df_feat['Vibration_Pressure_Product'] = df_feat['Vibration_mm_s'] * df_feat['Pressure_bar']
    
    # 6. Mechanical Fatigue Index (Torque * Tool Wear)
    df_feat['Torque_Wear_Product'] = df_feat['Torque_Nm'] * df_feat['Tool_Wear_min']
    
    return df_feat

def process_and_save_features(data_dir=os.path.join('data', 'raw')):
    raw_path = os.path.join(data_dir, 'machine_sensor_data.csv')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw sensor dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    df_feat = add_telematic_features(df)
    
    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sensor_features_full.csv')
    df_feat.to_csv(out_path, index=False)
    
    print(f"[SUCCESS] Feature engineering complete. Features dataset saved to {out_path}")
    print(f"Total Columns: {df_feat.shape[1]}")
    print("New Engineered Features:", ['Temp_Diff_K', 'Thermal_Stress_Index', 'Power_kW', 
                                       'Tool_Wear_Speed_Ratio', 'Vibration_Pressure_Product', 'Torque_Wear_Product'])
    return df_feat

if __name__ == '__main__':
    process_and_save_features()
