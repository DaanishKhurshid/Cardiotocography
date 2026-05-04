import streamlit as st
import joblib
import numpy as np

# Load your model and scaler
model = joblib.load('rf_best_model.pkl')
scaler = joblib.load('ctg_scaler.pkl')

st.set_page_config(page_title="CTG Fetal Health Monitor", layout="wide")
st.title("👶 Fetal Health Classification Tool")

# Organize inputs into columns to make it look professional
col1, col2, col3 = st.columns(3)

with col1:
    lb = st.number_input("Baseline Heart Rate (LB)", 100, 200, 130)
    ac = st.number_input("Accelerations (AC)", 0.0, 1.0, 0.0, format="%.3f")
    fm = st.number_input("Fetal Movement (FM)", 0.0, 1.0, 0.0, format="%.3f")
    uc = st.number_input("Uterine Contractions (UC)", 0.0, 1.0, 0.0, format="%.3f")
    dl = st.number_input("Light Decelerations (DL)", 0.0, 1.0, 0.0, format="%.3f")
    ds = st.number_input("Severe Decelerations (DS)", 0.0, 1.0, 0.0, format="%.3f")
    dp = st.number_input("Prolonged Decelerations (DP)", 0.0, 1.0, 0.0, format="%.3f")

with col2:
    astv = st.number_input("ASTV (%)", 0, 100, 50)
    mstv = st.number_input("MSTV (Mean)", 0.0, 10.0, 1.0)
    altv = st.number_input("ALTV (%)", 0, 100, 10)
    mltv = st.number_input("MLTV (Mean)", 0.0, 100.0, 10.0)
    width = st.number_input("Histogram Width", 0, 200, 50)
    min_val = st.number_input("Histogram Min", 0, 200, 50)
    max_val = st.number_input("Histogram Max", 0, 300, 150)

with col3:
    nmax = st.number_input("Histogram Nmax", 0, 100, 5)
    nzeros = st.number_input("Histogram Nzeros", 0, 10, 0)
    mode = st.number_input("Histogram Mode", 50, 250, 130)
    mean = st.number_input("Histogram Mean", 50, 250, 130)
    median = st.number_input("Histogram Median", 50, 250, 130)
    variance = st.number_input("Histogram Variance", 0, 300, 10)
    tendency = st.selectbox("Histogram Tendency", [-1, 0, 1], index=1)

if st.button("Generate Diagnostic Prediction"):
    # Must be in the EXACT same order as your 'features' list from the notebook
    input_data = np.array([[lb, ac, fm, uc, dl, ds, dp, astv, mstv, altv, mltv, 
                            width, min_val, max_val, nmax, nzeros, mode, mean, 
                            median, variance, tendency]])
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    # Visual Output
    if prediction == 1.0:
        st.success("Result: NORMAL ✅")
    elif prediction == 2.0:
        st.warning("Result: SUSPECT ⚠️")
    else:
        st.error("Result: PATHOLOGIC 🚨")
--------------------------------------------------------------------------