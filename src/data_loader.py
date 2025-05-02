import pandas as pd
import joblib
from configuration import MODEL_PATH, DATA_PATH

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
    return joblib.load(MODEL_PATH)