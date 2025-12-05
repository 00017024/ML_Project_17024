import streamlit as st
import pandas as pd
import joblib
import numpy as np

def apply_feature_engineering(df):
    df = df.copy()

    # Column normalization
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fix zero cholesterol
    if (df['cholesterol'] == 0).sum() > 0:
        chol_med = df[df['cholesterol'] > 0]['cholesterol'].median()
        df.loc[df['cholesterol'] == 0, 'cholesterol'] = chol_med

    # Fix zero blood pressure if exists
    if 'trestbps' in df.columns and (df['trestbps'] == 0).sum() > 0:
        bp_med = df[df['trestbps'] > 0]['trestbps'].median()
        df.loc[df['trestbps'] == 0, 'trestbps'] = bp_med

    # Age groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[-np.inf, 40, 55, 70, np.inf], 
        labels=[0,1,2,3]
    ).astype("Int64")

    # Age-cholesterol interaction
    df['age_chol_interaction'] = df['age'] * df['cholesterol'] / 1000

    # Heart rate reserve
    if 'max_heart_rate' in df.columns:
        df['heart_rate_reserve'] = df['max_heart_rate'] - (220 - df['age'])
    elif 'max heart rate' in df.columns:
        df['heart_rate_reserve'] = df['max heart rate'] - (220 - df['age'])

    # BP categories
    df['bp_category'] = pd.cut(
        df['trestbps'], 
        bins=[-np.inf, 120, 140, np.inf], 
        labels=[0,1,2]
    ).astype("Int64")

    # Cholesterol risk
    df['chol_risk'] = pd.cut(
        df['cholesterol'], 
        bins=[-np.inf, 200, 240, np.inf], 
        labels=[0,1,2]
    ).astype("Int64")

    return df


pipeline = joblib.load("model_pipeline.pkl")

# Title
st.title("Heart Attack Risk Prediction App")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Data Overview", "Predict"])

if page == "Home":
    st.title("Heart Attack Prediction App")
    st.write("Use the sidebar to explore data or make predictions.")


elif page == "Data Overview":
    st.title("Dataset Preview")
    df = pd.read_csv("data/Heart_Attack.csv")
    st.write("First 10 rows of dataset:")
    st.dataframe(df.head())

    st.write("Summary statistics:")
    st.write(df.describe())


elif page == "Predict":
    st.title("Risk Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 220, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=yes, 0=no)", [0, 1])
    restecg = st.number_input("Resting ECG (0–2)", 0, 2, 1)
    maxhr = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina (1=yes)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.number_input("Slope (0–2)", 0, 2, 1)

    user_input = {
        "age": age,
        "sex": sex,
        "chest_pain_type": cp,
        "trestbps": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": fbs,
        "resting_ecg": restecg,
        "max_heart_rate": maxhr,
        "exercise_angina": exang,
        "oldpeak": oldpeak,
        "st_slope": slope
    }

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = pipeline.predict(input_df)[0]
        if prediction == 1:
            st.error("High risk of heart disease")
        else:
            st.success("Low risk of heart disease")
