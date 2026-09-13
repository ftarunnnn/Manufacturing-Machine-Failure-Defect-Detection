# 🏭 Manufacturing — Machine Failure & Defect Detection System

An end-to-end industrial quality control and predictive maintenance intelligence platform combining **Tabular Machine Learning** (IoT machine failure prediction) and **Deep Computer Vision** (PyTorch CNN product defect classification).

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

---

## 📌 Project Overview
Modern smart factories require continuous monitoring of both operational telematics and visual product quality. This system unifies:
1. **Predictive Maintenance Engine (ML)**: Evaluates IoT telematic sensor readings (temperature, pressure, vibration, rotational speed, torque, tool wear) to predict machine failure risk before breakdown occurs.
2. **Visual Inspection Engine (CNN)**: Analyzes optical product surface scans to classify micro-defects into 5 categories (`Normal`, `Crack`, `Scratch`, `Dent`, `Surface Defect`).
3. **Unified Executive Dashboard**: An interactive Streamlit monitoring suite featuring real-time diagnostic reporting, high-recall alert systems, model telemetry, and interactive inspection tools.

---

## 🏗️ 10-Phase Project Architecture

- [x] **Phase 1 — Problem Definition**: Dual objective specification & architectural flow design.
- [x] **Phase 2 — Data Collection**: Telematic sensor dataset & multi-class defect image rendering pipeline.
- [x] **Phase 3 — Data Preprocessing**: Missing value imputation, outlier filtering, scaling, image normalization & PyTorch DataLoaders.
- [x] **Phase 4 — EDA & Data Analysis**: Sensor distribution analysis, correlation heatmaps, class balance checks, image dimensions breakdown.
- [x] **Phase 5 — Feature Engineering**: Temperature-pressure ratios, vibration rolling statistics, thermal stress indices, tool wear accumulation metrics.
- [x] **Phase 6 — ML Model Development**: Random Forest & XGBoost / Gradient Boosting classification pipelines tuned for high-recall safety.
- [x] **Phase 7 — CNN Model Development**: Custom PyTorch Deep CNN architecture trained on 5-class visual product defect images.
- [x] **Phase 8 — Model Evaluation**: Confusion matrices, ROC-AUC curves, Precision-Recall tradeoffs, classification metrics.
- [x] **Phase 9 — Integrated Inference**: Joint dual-inference engine delivering unified failure risk %, defect confidence, and operational recommendations.
- [x] **Phase 10 — Deployment & Dashboard**: Full interactive Streamlit monitoring dashboard with live inspection tools and report exports.

---

## ⚡ Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/ftarunnnn/Manufacturing-Machine-Failure-Defect-Detection.git
cd Manufacturing-Machine-Failure-Defect-Detection

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# 1. Generate Datasets
python data/generate_sensor_data.py
python data/generate_image_data.py

# 2. Train Models
python src/models/train_ml_model.py
python src/models/train_cnn_model.py

# 3. Launch Dashboard
streamlit run app.py
```

---

## 📊 Sample Output Report
```
============================================================
              MANUFACTURING MONITORING REPORT               
============================================================
[MACHINE STATUS]   : FAILURE RISK DETECTED
Failure Probability: 87.4%
Primary Failure    : Heat Dissipation Failure

[PRODUCT QUALITY]  : DEFECTIVE PRODUCT
Defect Type        : Surface Crack
Defect Confidence  : 94.2%

[RECOMMENDED ACTION]: QUARANTINE BATCH & HALT LINE FOR THERMAL MAINTENANCE
============================================================
```