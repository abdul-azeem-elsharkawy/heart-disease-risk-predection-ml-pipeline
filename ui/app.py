import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="Heart Disease Risk - ML App", page_icon="❤️", layout="centered")

st.title("Heart Disease Risk Prediction")
st.write("Provide your health indicators to get a risk prediction from the trained ML model.")

# Try to load the trained pipeline (preprocessing + model)
MODEL_PATH = Path(r"D:\Books and Courses\Machine Learning\Sprints x Microsoft Summer Camp - AI and Machine Learning\Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset (Graduation Project)\Heart_Disease_Project\models\final_model_(full_pipeline).pkl")
SCHEMA_PATH = Path(r"D:\Books and Courses\Machine Learning\Sprints x Microsoft Summer Camp - AI and Machine Learning\Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset (Graduation Project)\Heart_Disease_Project\ui\feature_schema.json")

pipeline = None
if MODEL_PATH.exists():
    try:
        pipeline = joblib.load(MODEL_PATH)
        st.success("Loaded trained model pipeline.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Load schema if available (created in preprocessing notebook)
schema = None
if SCHEMA_PATH.exists():
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        st.warning(f"Could not read feature_schema.json: {e}")

# Fallback to common UCI Cleveland columns if schema isn't present
fallback_schema = {
    "age": {"type": "numeric"},
    "sex": {"type": "categorical", "values": [0, 1]},
    "cp": {"type": "categorical", "values": [1, 2, 3, 4]},
    "trestbps": {"type": "numeric"},
    "chol": {"type": "numeric"},
    "fbs": {"type": "categorical", "values": [0, 1]},
    "restecg": {"type": "categorical", "values": [0, 1, 2]},
    "thalach": {"type": "numeric"},
    "exang": {"type": "categorical", "values": [0, 1]},
    "oldpeak": {"type": "numeric"},
    "slope": {"type": "categorical", "values": [1, 2, 3]},
    "ca": {"type": "categorical", "values": [0, 1, 2, 3]},
    "thal": {"type": "categorical", "values": [0, 3, 6, 7]}
}

feature_schema = schema.get("features") if isinstance(schema, dict) and "features" in schema else fallback_schema

# Build input form
with st.form("input_form"):
    inputs = {}
    for feat, meta in feature_schema.items():
        label = meta.get("label", feat) 
        if meta.get("type") == "numeric":
            val = st.number_input(label, value=0.0, step=0.1, format="%.2f")
        elif meta.get("type") == "categorical":
            values = meta.get("values", [])
            if values:
                val = st.selectbox(label, options=values, index=0)
            else:
                val = st.number_input(f"{label} (categorical)", value=0, step=1)
        else:
            val = st.number_input(label, value=0.0, step=0.1, format="%.2f")
        inputs[feat] = val


    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([inputs])
    if pipeline is None:
        st.error("Model pipeline not found. Train and save your model to 'models/final_model_(full_pipeline).pkl'.")
    else:
        try:
            proba = None
            if hasattr(pipeline, "predict_proba"):
                proba = float(pipeline.predict_proba(df)[0, 1])
            pred = int(pipeline.predict(df)[0])
            st.subheader("🔮 Prediction Result")

            # Show result
            if pred == 1:
                st.success("✅ High Risk of Heart Disease Detected")
            else:
                st.info("🫀 Low Risk of Heart Disease")

            # Show probability in the form of a progress bar
            if proba is not None:
                st.markdown("### Probability of Heart Disease")
                st.progress(proba)  # bar
                st.write(f"**{proba*100:.1f}%** chance of class = 1")

        except Exception as e:
            st.exception(e)
