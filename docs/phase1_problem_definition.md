# Phase 1 — Problem Definition & System Objectives

## Executive Summary
In modern smart manufacturing, equipment downtime and defective product output lead to substantial financial losses, production delays, and safety risks. This project establishes an **AI-Powered Manufacturing Monitoring & Quality Control System** integrating two core capabilities:
1. **Predictive Maintenance (Machine Learning)**: Predict machine failures before they occur using real-time telematic sensor readings.
2. **Visual Quality Inspection (Deep Learning / Computer Vision)**: Automatically detect and classify surface defects on manufactured products from high-resolution optical images.

---

## 1. System Objectives

### ML Objective — Machine Failure Prediction
- **Task**: Binary Classification & Failure Risk Estimation.
- **Input**: Telematic sensors attached to manufacturing machinery (Temperature, Pressure, Vibration, Rotational Speed, Torque, Operating Hours, Tool Wear, Ambient Temperature).
- **Output**: 
  - `Machine_Status`: Normal (0) vs Failure Risk (1).
  - `Failure_Probability`: Continuous confidence percentage ($[0\%, 100\%]$).
  - `Failure_Type`: Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, or Random Failure.

### DL Objective — Visual Product Defect Detection
- **Task**: Multi-Class Image Classification.
- **Input**: Optical RGB images of manufactured product surfaces ($128 \times 128$ resolution).
- **Output**:
  - `Product_Status`: Normal vs Defective.
  - `Defect_Type`: `Normal`, `Crack`, `Scratch`, `Dent`, `Surface Defect`.
  - `Defect_Confidence`: Softmax class probability ($[0\%, 100\%]$).

---

## 2. Key Target Metrics
In manufacturing environments, **Recall** for machine failure and product defect detection is the single most critical metric. Missing a real failure (False Negative) can cause severe machine breakdown or catastrophic product failure in customer applications.

- **Primary Target**: Recall $\ge 90\%$ for failure and defect classes.
- **Secondary Targets**: Precision $\ge 85\%$, F1-Score $\ge 88\%$, ROC-AUC $\ge 0.92$.

---

## 3. Integrated Output Format
When an inspection cycle runs on a machine unit and product sample, the combined inference engine generates a unified report:

```json
{
  "timestamp": "2026-09-13T21:20:00Z",
  "machine_health": {
    "status": "Failure Risk",
    "failure_probability": 0.87,
    "primary_failure_mode": "Heat Dissipation Failure",
    "risk_level": "CRITICAL"
  },
  "product_quality": {
    "status": "Defective",
    "defect_type": "Surface Crack",
    "defect_confidence": 0.94,
    "inspection_result": "REJECT"
  },
  "overall_action": "QUARANTINE PRODUCT & SCHEDULE IMMEDIATE THERMAL MAINTENANCE"
}
```

---

## 4. System Architecture Flow

```
                      Manufacturing Telematics & Vision Stream
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ↓                                                       ↓
      IoT Machine Sensor Data                                 Product Surface Image
(Temp, Pressure, Vibration, RPM, Torque)                 (Optical Inspection Camera)
             ↓                                                       ↓
  Preprocessors & Feature Extractor                       Resizer, Normalizer & Augmenter
             ↓                                                       ↓
Random Forest / XGBoost Classifier                       PyTorch Deep CNN Classifier
             ↓                                                       ↓
  Machine Failure Risk (87%)                              Defect Class: Crack (94%)
             └───────────────────────────┬───────────────────────────┘
                                         ↓
                            Unified Inspection Engine
                                         ↓
                     Interactive Streamlit Executive Dashboard
```
