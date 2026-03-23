# Smart Crop Stress Advisor

This project predicts plant health status from sensor data and turns the prediction into a usable crop advisory workflow.

Instead of stopping at "Healthy / Moderate Stress / High Stress", it now adds:

- a hybrid ML model for plant stress prediction
- a 0-100 risk score for triage
- sensor-level stress driver analysis
- field recommendations for what to fix first
- a Streamlit dashboard for single and batch diagnosis
- monitoring log generation for operational use

## Why This Version Is Stronger

This is no longer just a notebook classification exercise. It is framed as an IoT crop stress early-warning system:

- sensor readings are converted into plant health predictions
- the model output is combined with healthy-range deviation
- the system explains which sensor readings are pushing the plant into stress
- the app can prioritize a full CSV of plants by risk

That is a much better interview story than "I trained Random Forest and SVM on a CSV".

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- SHAP
- Streamlit

## Project Structure

- `data/`: raw and processed datasets
- `models/`: saved scaler, encoder, and trained models
- `notebooks/hybrid/`: training and analysis pipeline
- `outputs/plots/`: charts and evaluation plots
- `smart_advisor.py`: reusable advisory and risk-scoring engine
- `app.py`: Streamlit dashboard

## Streamlit App

Run the dashboard:

```bash
streamlit run app.py
```

The app provides:

- `Single Plant Check`: manual sensor entry with prediction, risk score, top drivers, and recommended actions
- `Batch Triage`: upload a CSV and rank plants by risk
- `Project Story`: resume bullets and architecture framing

Predictions are logged to:

```text
outputs/monitoring/prediction_log.csv
```

<<<<<<< HEAD
## Notebook Run Order
=======
## Notebook Run Order:-
>>>>>>> a6640674860749d930e0b20530390417987b96a2

1. `01_data_loading.ipynb`
2. `02_preprocessing.ipynb`
3. `03_eda.ipynb`
4. `04_random_forest.ipynb`
5. `05_svm.ipynb`
6. `06_hybrid_model.ipynb`
7. `07_model_comparison.ipynb`
8. `08_prediction.ipynb`
9. `09_risk_score_and_explainability.ipynb`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset:-

Expected raw dataset file:

```text
data/plant_health_data.csv
```

Main features used by the advisory system:

- Soil_Moisture
- Ambient_Temperature
- Soil_Temperature
- Humidity
- Light_Intensity
- Soil_pH
- Nitrogen_Level
- Phosphorus_Level
- Potassium_Level
- Chlorophyll_Content
- Electrochemical_Signal

<<<<<<< HEAD
## Resume Framing
=======
## Resume Framing:-
>>>>>>> a6640674860749d930e0b20530390417987b96a2

You can describe this project as:

> Built a smart agriculture early-warning system that predicts plant stress from IoT sensor data, scores crop risk, explains stress drivers, and recommends corrective field actions through a Streamlit dashboard.
