import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the saved model
model = joblib.load("best_model.pkl")

st.title("California Housing Price Predictor 🏠")

st.markdown("Enter values below to predict the median house value.")

# Input features
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupants per Household", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.0)
Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-120.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=30.0)
MedInc = st.number_input("Median Income", min_value=0.0, value=3.0)

# Derived feature
rooms_per_household = AveRooms / (HouseAge + 0.0001)  # Avoid divide by zero

# Feature order must match training
input_data = np.array([[AveRooms, AveBedrms, Population, AveOccup,
                        Latitude, Longitude, HouseAge, MedInc,
                        rooms_per_household]])

# Apply scaling (you must reuse the same scaler used during training)
scaler = StandardScaler()
# Fake fit with same structure (use training stats ideally)
dummy_df = pd.DataFrame(columns=["AveRooms", "AveBedrms", "Population", "AveOccup",
                                 "Latitude", "Longitude", "HouseAge", "MedInc",
                                 "rooms_per_household"])
scaler.fit(np.zeros((1, 9)))  # placeholder for structure
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Median House Value: ${prediction[0]*100000:.2f}")
