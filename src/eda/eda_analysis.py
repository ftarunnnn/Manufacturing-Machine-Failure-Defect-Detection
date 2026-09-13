import os
import json
import numpy as np
import pandas as pd
from PIL import Image

def run_tabular_eda(df_path=os.path.join('data', 'raw', 'machine_sensor_data.csv')):
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Raw dataset not found at {df_path}")
        
    df = pd.read_csv(df_path)
    sensor_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 
                   'Torque_Nm', 'Tool_Wear_min', 'Pressure_bar', 'Vibration_mm_s']
    
    # 1. Summary Statistics
    desc = df[sensor_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    desc.columns = ['mean', 'std', 'min', 'median', 'max']
    stats_dict = desc.to_dict(orient='index')
    
    # 2. Correlation with Target
    corr = df[sensor_cols + ['Machine_Failure']].corr()['Machine_Failure'].drop('Machine_Failure').to_dict()
    
    # 3. Class Imbalance
    class_balance = df['Machine_Failure'].value_counts(normalize=True).to_dict()
    class_counts = df['Machine_Failure'].value_counts().to_dict()
    
    # 4. Failure vs Sensor Values (Grouped Means)
    failure_group_means = df.groupby('Machine_Failure')[sensor_cols].mean().to_dict()
    
    # 5. Failure Type Distribution
    failure_types = df['Failure_Type'].value_counts().to_dict()
    
    tabular_report = {
        'total_records': len(df),
        'sensor_statistics': stats_dict,
        'correlation_with_failure': corr,
        'class_balance_ratio': class_balance,
        'class_counts': class_counts,
        'sensor_means_by_failure_status': failure_group_means,
        'failure_type_breakdown': failure_types
    }
    
    print("[SUCCESS] Tabular Sensor EDA Complete.")
    print(f"Total Records: {len(df)}")
    print(f"Machine Failure Rate: {class_balance.get(1, 0)*100:.2f}%")
    return tabular_report

def run_image_eda(img_dir=os.path.join('data', 'raw', 'images')):
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found at {img_dir}")
        
    classes = ['normal', 'crack', 'scratch', 'dent', 'surface_defect']
    class_counts = {}
    image_shapes = []
    
    for c in classes:
        c_path = os.path.join(img_dir, c)
        if os.path.exists(c_path):
            files = [f for f in os.listdir(c_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[c] = len(files)
            for f in files[:10]: # Sample shapes
                with Image.open(os.path.join(c_path, f)) as img:
                    image_shapes.append(img.size)
        else:
            class_counts[c] = 0
            
    unique_shapes = list(set(image_shapes))
    
    image_report = {
        'defect_class_counts': class_counts,
        'total_images': sum(class_counts.values()),
        'sample_dimensions_wh': unique_shapes[0] if unique_shapes else (128, 128)
    }
    
    print("[SUCCESS] Image Defect EDA Complete.")
    print(f"Total Defect Images: {image_report['total_images']}")
    print("Class Counts:", class_counts)
    return image_report

def generate_eda_report():
    tab_report = run_tabular_eda()
    img_report = run_image_eda()
    
    full_report = {
        'tabular_sensor_eda': tab_report,
        'product_image_eda': img_report
    }
    
    out_dir = 'reports'
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'eda_summary.json')
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
        
    print(f"[SUCCESS] Full EDA Summary saved to {report_path}")

if __name__ == '__main__':
    generate_eda_report()
