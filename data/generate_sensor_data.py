import os
import numpy as np
import pandas as pd

def generate_machine_sensor_dataset(num_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base Feature Generation
    air_temp = np.random.uniform(295.0, 304.5, num_samples) # Ambient temperature in Kelvin
    # Process temp is correlated with air temp + machine thermal load
    process_temp = air_temp + np.random.uniform(10.0, 15.0, num_samples)
    
    # Rotational speed (RPM)
    rotational_speed = np.random.normal(1500, 250, num_samples)
    rotational_speed = np.clip(rotational_speed, 1100, 2800)
    
    # Torque (Nm) - inversely correlated with RPM + noise
    torque = 70.0 - (rotational_speed - 1100) * 0.02 + np.random.normal(0, 10, num_samples)
    torque = np.clip(torque, 10.0, 90.0)
    
    # Tool wear (minutes)
    tool_wear = np.random.uniform(0, 240, num_samples)
    
    # Hydraulic/Pneumatic Pressure (bar)
    pressure = np.random.normal(5.5, 1.2, num_samples)
    pressure = np.clip(pressure, 2.0, 10.0)
    
    # Vibration (mm/s)
    vibration = np.random.gamma(shape=2.0, scale=1.5, size=num_samples)
    vibration = np.clip(vibration, 0.5, 15.0)
    
    # 2. Physics-Inspired Failure Modes
    failure_type = []
    machine_failure = []
    
    for i in range(num_samples):
        # Calculate key mechanical stress indicators
        power_kW = (2 * np.pi * rotational_speed[i] * torque[i]) / 60000.0
        temp_diff = process_temp[i] - air_temp[i]
        
        # Mode A: Tool Wear Failure (TWF)
        twf = (tool_wear[i] >= 200) and (np.random.rand() < 0.65 or torque[i] > 60)
        
        # Mode B: Heat Dissipation Failure (HDF)
        hdf = (temp_diff < 8.8) and (rotational_speed[i] < 1380)
        
        # Mode C: Power Failure (PWF)
        pwf = (power_kW < 3.5 or power_kW > 9.0) and (np.random.rand() < 0.70)
        
        # Mode D: Overstrain Failure (OSF)
        osf = (tool_wear[i] * torque[i] > 11500) or (vibration[i] > 10.5)
        
        # Mode E: Random Failure (RNF)
        rnf = np.random.rand() < 0.005
        
        if twf:
            failure_type.append('Tool Wear Failure')
            machine_failure.append(1)
        elif hdf:
            failure_type.append('Heat Dissipation Failure')
            machine_failure.append(1)
        elif pwf:
            failure_type.append('Power Failure')
            machine_failure.append(1)
        elif osf:
            failure_type.append('Overstrain Failure')
            machine_failure.append(1)
        elif rnf:
            failure_type.append('Random Failure')
            machine_failure.append(1)
        else:
            failure_type.append('No Failure')
            machine_failure.append(0)
            
    df = pd.DataFrame({
        'UDI': np.arange(1, num_samples + 1),
        'Air_Temperature_K': np.round(air_temp, 2),
        'Process_Temperature_K': np.round(process_temp, 2),
        'Rotational_Speed_RPM': np.round(rotational_speed, 1),
        'Torque_Nm': np.round(torque, 2),
        'Tool_Wear_min': np.round(tool_wear, 0).astype(int),
        'Pressure_bar': np.round(pressure, 2),
        'Vibration_mm_s': np.round(vibration, 2),
        'Machine_Failure': machine_failure,
        'Failure_Type': failure_type
    })
    
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'machine_sensor_data.csv')
    df.to_csv(file_path, index=False)
    
    print(f"[SUCCESS] Machine sensor dataset generated: {file_path}")
    print(f"Total Records: {len(df)}")
    print(f"Failures: {df['Machine_Failure'].sum()} ({df['Machine_Failure'].mean()*100:.2f}%)")
    print("Failure Type Breakdown:")
    print(df['Failure_Type'].value_counts())
    return df

if __name__ == '__main__':
    generate_machine_sensor_dataset()
