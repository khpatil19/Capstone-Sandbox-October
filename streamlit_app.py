import pandas as pd
from joblib import load
import streamlit as st

# Load the model from the file
rf_pipeline = load('C:/Users/khpat/Desktop/Desktop Files/LabelMaster/random_forest_regressor.joblib')


st.title("Capstone model demo")
st.write(
    "Testing changes khpatil"
)
