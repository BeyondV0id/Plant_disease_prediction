# Plant Disease Prediction

This project predicts plant health status (Healthy, Moderate Stress, High Stress) using sensor data and machine learning (Random Forest & SVM).

## Structure

- **data/**: Raw and processed datasets
- **notebooks/**: Step-by-step Jupyter notebooks
- **models/**: Saved ML models and scaler
- **outputs/**: Results and plots
- 

## Run Order

1. 01_data_loading.ipynb
2. 02_preprocessing.ipynb
3. 03_eda.ipynb
4. 04_random_forest.ipynb
5. 05_svm.ipynb
6. 06_hybrid_model.ipynb
7. 07_model_comparison.ipynb
8. 08_prediction.ipynb
9. 09_risk_score_and_explainability.ipynb

## Novelty Extension

The project now includes an advanced notebook for:

- Continuous disease risk scoring (0-100)
- Alert generation for decision support
- SHAP-based sensor contribution analysis
- Timestamp-based temporal health trend tracking

## Setup

Install requirements:

```
pip install -r requirements.txt
```

## Dataset

Place your original CSV as `data/plant_health_data.csv` before running the notebooks.
# plant_disease_detection
