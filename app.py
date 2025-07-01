import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("data/Clean_Dataset.csv")
df.drop(columns=['Unnamed: 0', 'flight'], inplace=True)
df['price'] = np.log1p(df['price'])

df = pd.get_dummies(df, columns=[
    'airline', 'source_city', 'destination_city', 'departure_time',
    'arrival_time', 'stops', 'class'
], drop_first=True)

# Define feature columns
feature_prefixes = ['airline_', 'source_city_', 'destination_city_', 'class_']
X = df[[col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]]
y = df['price']

# Train Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# UI
st.title("✈️ Flight Price Prediction App")

airline = st.selectbox("Select Airline", ['Indigo', 'Air India', 'SpiceJet', 'Vistara', 'GO_FIRST', 'AirAsia'])
source_city = st.selectbox("Select Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai'])
destination_city = st.selectbox("Select Destination City", ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai'])
flight_class = st.selectbox("Select Class", ['Economy', 'Business'])

if st.button("Predict Price"):
    # Create input vector
    user_df = pd.DataFrame([0]*X.shape[1], index=X.columns).T

    for col in X.columns:
        if f"airline_{airline}" == col:
            user_df[col] = 1
        if f"source_city_{source_city}" == col:
            user_df[col] = 1
        if f"destination_city_{destination_city}" == col:
            user_df[col] = 1
        if f"class_{flight_class}" == col:
            user_df[col] = 1

    log_pred = model.predict(user_df)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"Estimated Flight Price: ₹{round(price_pred, 2)}")
