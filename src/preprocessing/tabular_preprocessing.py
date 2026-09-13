import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_raw_sensor_data(file_path=os.path.join('data', 'raw', 'machine_sensor_data.csv')):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw sensor dataset not found at {file_path}. Run data/generate_sensor_data.py first.")
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded raw sensor dataset: {df.shape}")
    return df

def preprocess_tabular_data(df, target_col='Machine_Failure'):
    # 1. Handle Missing Values & Duplicates
    df = df.drop_duplicates().copy()
    num_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
                'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s']
    
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Outlier Clipping (3 * IQR limits to avoid extreme noise corruption)
    for col in num_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 3.0 * iqr
        upper_bound = q75 + 3.0 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    return df

def split_and_scale_tabular_data(df, target_col='Machine_Failure', test_size=0.2, val_size=0.1, random_seed=42):
    feature_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
                    'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s']
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Train / Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Train / Val split
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_seed, stratify=y_train_val
    )
    
    # Fit StandardScaler strictly on X_train to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    
    # Save Scaler
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] StandardScaler fitted and saved to {scaler_path}")
    
    # Combine back targets for saving processed splits
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    val_df = pd.concat([X_val_scaled, y_val], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)
    
    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, 'sensor_train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'sensor_val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'sensor_test.csv'), index=False)
    
    print(f"[SUCCESS] Processed tabular splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df, scaler

if __name__ == '__main__':
    raw_df = load_raw_sensor_data()
    clean_df = preprocess_tabular_data(raw_df)
    split_and_scale_tabular_data(clean_df)
