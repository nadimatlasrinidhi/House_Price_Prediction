import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/models/house_prediction.pkl")

# Load dataset
df = pd.read_csv("data/housing.csv")

st.title("🏠 House Price Prediction")

# Location dropdown
location = st.selectbox(
    "Select Location",
    sorted(df["location"].unique())
)

# Inputs
area = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=10000,
    value=1500
)

bhk = st.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

bath = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "location": [location],
        "total_sqft": [area],
        "bhk": [bhk],
        "bath": [bath]
    })

    price = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {price:,.0f}")