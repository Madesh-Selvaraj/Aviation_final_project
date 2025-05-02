# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and mappings
model = joblib.load("model.pkl")
options = joblib.load("options.pkl")

origin_map = options["origin_map"]
dest_map = options["dest_map"]

st.set_page_config(page_title="Flight Delay Predictor")
st.title("✈️ Flight Delay Predictor")

# User input: select from real values
origin = st.selectbox("Origin Airport", options["origin_options"])
dest = st.selectbox("Destination Airport", options["dest_options"])
hour = st.slider("Departure Hour", 0, 23, 12)

if st.button("Predict"):
    input_data = np.array([[origin_map[origin], dest_map[dest], hour]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'Delayed' if pred else 'On Time'}")
    st.write(f"### Probability of Delay: {prob:.2%}")