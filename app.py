import streamlit as st
import numpy as np
import joblib

# Load models
rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

st.title("❤️ Heart Disease Prediction App")
st.subheader("Choose model and enter patient information")

# Select model
model_choice = st.selectbox("Select Model", 
    ["Random Forest", "Decision Tree", "Logistic Regression"])

# Inputs (matching the model's 15 features)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=130)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.5)
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# Engineered features
chol_age_ratio = chol / age if age != 0 else 0
risk_score = cp + exang + slope + ca  # Adjust if your original formula is different

if age < 40:
    age_group = 0
elif age < 60:
    age_group = 1
else:
    age_group = 2

# Combine all inputs in correct order
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal,
                        chol_age_ratio, risk_score,age_group]])

# Predict button
if st.button("Predict"):
    # Select the model
    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Decision Tree":
        model = dt_model
    else:
        model = lr_model

    # Make prediction
    prediction = model.predict(input_data)

    # Output result
    if prediction[0] == 1:
        st.error("🚨 High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")
