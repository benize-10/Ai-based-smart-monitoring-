# Project Report: AI-Based Smart Inverter Performance Monitoring

> Editable draft. Fill in your own details and replace placeholders with actual results. Do not invent metrics.

## Title page
- Student: [Your Name]
- Register number: [Register Number]
- Department / College: [Department and College]
- Guide: [Guide Name]
- Academic year: [Academic Year]

## Abstract
This project demonstrates a software workflow for analysing inverter telemetry using machine learning. Voltage, current, and temperature features are used by a Random Forest regressor to estimate efficiency and a Random Forest classifier to categorise operating status as normal, degraded, or abnormal. A Streamlit dashboard displays telemetry trends and predictions. The included data is synthetic and is only suitable for demonstrating the code workflow; independent validation with real data is required for real-world conclusions.

## 1. Introduction
Inverter performance can vary with load, temperature, operating conditions, and component degradation. Data-driven monitoring can help summarise telemetry and flag records for review. It must not replace hardware protection or qualified engineering assessment.

## 2. Problem statement and objectives
Develop a prototype that collects or loads input/output voltage, current, temperature, and efficiency data; predicts efficiency or performance class; explores poor-performance patterns; and displays results on a dashboard.

Objectives:
1. Validate and explore telemetry.
2. Train regression and classification models.
3. Evaluate with MAE, RMSE, precision, recall, F1-score, and a confusion matrix.
4. Visualise trends and sample predictions.

## 3. System architecture
CSV telemetry → data validation → feature selection → train/test split → Random Forest regressor and classifier → evaluation → Streamlit dashboard.

## 4. Dataset
Source: [state the actual source, or clearly say synthetic].
Rows: [actual count]. Sampling interval: [interval].
Features: input voltage, output voltage, input current, output current, temperature.
Targets: efficiency percentage and performance status.
Describe units, sensor calibration, missing-value handling, data license, and class definitions.

## 5. Methodology
Check schema, missing values, outliers, units, and timestamp order. Split data before fitting models. Train a Random Forest regressor for efficiency and classifier for status. Report MAE/RMSE and per-class precision/recall/F1. Use a held-out test set; for time-series data, a time-based split is generally preferable.

## 6. Implementation
`src/generate_sample_data.py` creates synthetic demonstration data. `src/train.py` trains the models and saves metrics. `src/evaluate.py` exports evaluation results. `app.py` provides a Streamlit interface for trends, status distribution, efficiency, and predictions when model files exist.

## 7. Results
- Regression MAE: [actual result]
- Regression RMSE: [actual result]
- Classification accuracy: [actual result]
- Per-class precision/recall/F1: [actual results]
- Test set and evaluation method: [describe]
If only synthetic data was used, say so clearly. Synthetic scores are not evidence of field performance.

## 8. Limitations and safety
Sample data is synthetic; class labels are illustrative. Results are not validated for a specific inverter. The software is not a certified monitoring, control, or protection system. Do not use predictions for safety-critical decisions.

## 9. Future work
Collect documented telemetry, use time-aware validation, add relevant features such as load or power factor where available, compare algorithms, and add monitoring alerts after appropriate validation.

## 10. Conclusion
The prototype demonstrates telemetry analysis, efficiency regression, performance classification, evaluation, and dashboard visualisation. Conclusions must be limited to what the available data and independent testing support.

## References
[Add the real papers, dataset sources, equipment manuals, and documentation used.]
