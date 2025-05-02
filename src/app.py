import streamlit as st
from data_loader import load_data, load_model
from predictor import predict_incident
from visualizer import plot_confusion, plot_roc
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Flight Incident Predictor", layout="wide")

# Load data and model
data = load_data()
model = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["Incident Predictor", "Model Performance", "Data Exploration"])

# --- Incident Predictor ---
with tab1:
    st.header("Predict Flight Incident")
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("Select Origin", data["origin"].unique())
    with col2:
        destination = st.selectbox("Select Destination", data["destination"].unique())
    with col3:
        dep_time = st.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, value=700)

    if st.button("Predict Incident"):
        prediction, prob = predict_incident(model, origin, destination, dep_time)
        st.success(f"Prediction: {'Incident' if prediction else 'No Incident'} (Probability: {prob:.2f})")

# --- Model Performance ---
with tab2:
    st.header("Model Performance")
    y_true = data["incident"]
    X = data[["origin", "destination", "departure_time"]]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    st.pyplot(plot_confusion(y_true, y_pred))
    st.pyplot(plot_roc(y_true, y_proba))

# --- Data Exploration ---
with tab3:
    st.header("Explore Dataset")
    st.dataframe(data)
    st.map(data[["LAT", "LON"]]) if "LAT" in data.columns and "LON" in data.columns else st.info("No geo info to map.")
