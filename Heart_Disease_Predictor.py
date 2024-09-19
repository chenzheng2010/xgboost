import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = [
    "Length", "Width", "Area", "R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
Length = st.number_input("Length:", min_value=100, max_value=600, value=500)

# trestbps: numerical input
Width = st.number_input("Width:", min_value=50, max_value=200, value=120)

# thalach: numerical input
Area = st.number_input("Area:", min_value=41156.708, max_value=157665.5032, value=70958.9697)

# oldpeak: numerical input
R_mean = st.number_input("R_mean:", min_value=50, max_value=100, value=60)

# oldpeak: numerical input
R_std = st.number_input("R_std:", min_value=10.4538, max_value=28.6495, value=12.5809)

# oldpeak: numerical input
G_mean = st.number_input("G_mean:", min_value=107.2145, max_value=196.69, value=127.7122)

# oldpeak: numerical input
G_std = st.number_input("G_std:", min_value=8.2652, max_value=18.6073, value=16.1242)

# oldpeak: numerical input
B_mean = st.number_input("B_mean:", min_value=42.0448, max_value=135.5916, value=125.4943)

# oldpeak: numerical input
B_std = st.number_input("B_std:", min_value=10.5494, max_value=32.8174, value=11.9413)

# Process inputs and make predictions
feature_values = [Length, Width, Area, R_mean, R_std, G_mean, G_std, B_mean, B_std]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")