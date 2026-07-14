import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Get project root directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Load Model
# -----------------------------
model_path = os.path.join(BASE_DIR, "models", "house_prediction.pkl")
model = joblib.load(model_path)

# -----------------------------
# Load Dataset
# -----------------------------
data_path = os.path.join(BASE_DIR, "data", "housing.csv")
df = pd.read_csv(data_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hyderabad House Price Prediction")

st.title("🏠 Hyderabad House Price Prediction")
st.write("Enter the house details to predict the estimated price.")

# -----------------------------
# Location
# -----------------------------
location = st.selectbox(
    "Select Location",
    sorted(df["location"].unique())
)

# -----------------------------
# Inputs
# -----------------------------
area = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=6000,
    value=1500,
    step=100
)

bhk = st.number_input(
    "BHK",
    min_value=1,
    max_value=6,
    value=3
)

bath = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=6,
    value=2
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "location": [location],
        "total_sqft": [area],
        "bhk": [bhk],
        "bath": [bath]
    })

    predicted_price = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {predicted_price:,.0f}")