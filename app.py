# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import random

# -------------------------------
# Load all models
# -------------------------------
MODEL_DIR = "models"

model_dict = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "log_reg.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl")),
    "XGBoost (Balanced)": joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
    "Hybrid Model": joblib.load(os.path.join(MODEL_DIR, "hybrid_model.pkl"))
}

# Feature names including Time
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Sample input data
random_samples = [
    [406, -2.312226542, 1.951992011, -1.609850732, 3.997905588, -0.522187865,
     -1.426545319, -2.537387306, 1.391657248, -2.770089277, -2.772272145,
     3.202033207, -2.899907388, -0.595221881, -4.289253782, 0.38972412,
     -1.14074718, -2.830055675, -0.016822468, 0.416955705, 0.126910559,
     0.517232371, -0.035049369, -0.465211076, 0.320198199, 0.044519167,
     0.177839798, 0.261145003, -0.143275875, 0],
    [4462, -2.303349568, 1.75924746, -0.359744743, 2.330243051, -0.821628328, -0.075787571,
     0.562319782, -0.399146578, -0.238253368, -1.525411627, 2.032912158, -6.560124295,
     0.022937323, 11.470101536, -0.698826069, -2.282193829, -4.781830856, -2.615664945,
     -1.334441067, -0.430021867, -0.294166318, -0.932391057, 0.172726296, -0.087329538,
     -0.156114265, -0.542627889, 0.039565989, -0.153028797, 239.93],
    [7519, 1.234235046, 3.019740421, -4.304596885, 4.73279513, 3.624200831, -1.357745663,
     1.713444988, -0.496358487, -1.28285782, -2.447469255, 2.101343865, -4.609628391,
     1.464377625, -6.079337193, -0.339237373, 2.581850954, 6.739384385, 3.042493178,
     -2.721853122, 0.009060836, -0.379068307, -0.704181032, -0.656804756, -1.632652957,
     1.488901448, 0.566797273, -0.010016223, 0.146792735, 1],
    [7558, -5.234481027, 4.435237562, -0.232470144, -2.28103297, -1.044720027, -1.069926818,
     0.480053017, -0.151234415, 5.382178101, 6.021094035, 3.01830673, -1.759176799,
     1.531517953, -0.419358652, -0.545440612, 0.066258226, -0.570133568, -0.073553889,
     -1.278336092, 2.500895496, -0.948289794, -0.498932635, 0.120434135, 0.433980356,
     0.416345372, 0.449141219, -0.363813123, -1.11826701, 0.77],
    [0.12, -1.1, 0.3, -0.6, 1.0, 0.5, 0.1, -0.9, 0.7, -1.2, -0.4, 0.3, -0.7, 1.2, -0.9,
     0.8, -0.1, -0.3, 0.5, -0.6, -0.7, 0.2, 0.1, -0.2, 0.3, 0.0, -0.4, 0.6, -0.5, 0.42],
    [0.04, 0.8, -0.2, 1.1, -0.9, -0.1, 0.3, 0.6, -0.7, 0.5, -0.8, 0.1, -0.3, 0.6, -0.2,
     -0.5, 0.3, 0.2, -0.4, 0.5, -0.1, -0.2, 0.3, 0.1, 0.2, -0.1, 0.4, -0.3, 0.2, -0.15],
    [0.89, -0.6, 0.1, -11, 11, 0.9, -0.7, 0.3, 0.2, -0.4, 0.7, -0.2, 0, -0.8, -11, 0.1,
     0.2, -0.5, 0.3, -0.3, 0.6, -0.4, 0.5, -0.6, 0.7, 0.1, -0.2, 0.3, -0.1, 0.87],
]

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 AI-Driven Credit Card Fraud Detection")

# Tabs
tabs = st.tabs(["🔍 Prediction", "📊 Model Comparison", "🧠 Explainability"])

# -------------------------------
# 🔍 Prediction Tab
# -------------------------------
with tabs[0]:
    st.header("Run Prediction")
    selected_model_name = st.selectbox("Choose a Model", list(model_dict.keys()))
    model = model_dict[selected_model_name]

    # Use Random Sample Button
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {feature: 0.0 for feature in feature_names}

    if st.button("Use Random Sample"):
        sample = random.choice(random_samples)
        st.session_state.user_inputs = dict(zip(feature_names, sample))

    st.subheader("Enter Transaction Details")
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", value=st.session_state.user_inputs[feature])

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ Fraud Detected (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {1 - proba:.2%})")

# -------------------------------
# 📊 Model Comparison Tab
# -------------------------------
with tabs[1]:
    st.header("Compare Model Performance (ROC-AUC)")

    auc_scores = {
        "Logistic Regression": 0.9634,
        "Random Forest": 0.9793,
        "XGBoost (Balanced)": 0.9918,
        "Hybrid Model": 0.9879,
        "XGBoost (Imbalanced)": 0.9412
    }

    df_auc = pd.DataFrame(list(auc_scores.items()), columns=["Model", "AUC Score"])
    df_auc = df_auc.sort_values("AUC Score", ascending=True)

    st.bar_chart(df_auc.set_index("Model"))

    with st.expander("📈 Show AUC Table"):
        st.table(df_auc.sort_values("AUC Score", ascending=False))

# -------------------------------
# 🧠 Explainability Tab (SHAP)
# -------------------------------
with tabs[2]:
    st.header("Explain Predictions with SHAP")

    shap_model = model_dict["XGBoost (Balanced)"]
    explainer = shap.Explainer(shap_model, feature_names=feature_names)

    shap_input = pd.DataFrame([input_data])[feature_names]
    shap_values = explainer(shap_input)

    st.subheader("SHAP Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, shap_input, show=False)
    st.pyplot(fig)

    st.info("Top features with highest impact on the XGBoost model's prediction.")
