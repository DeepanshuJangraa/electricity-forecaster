import datetime
import os
import pickle

import pandas as pd
import streamlit as st


# Load the saved model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.pkl")
    with open(model_path, "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("⚡ Electricity Consumption Forecaster")
st.write("Predict future energy demand based on time and historical usage.")

col1, col2 = st.columns(2)
with col1:
    pred_date = st.date_input("Select Date", datetime.date.today())
    pred_time = st.time_input("Select Time", datetime.time(12, 00))
with col2:
    lag_value = st.number_input("Yesterday's Consumption at this time (MW)", min_value=0.0, value=15000.0)

if st.button("Predict Consumption"):
    hour = pred_time.hour
    dayofweek = pred_date.weekday()
    month = pred_date.month
    year = pred_date.year
    dayofyear = pred_date.timetuple().tm_yday

    input_features = pd.DataFrame(
        [[hour, dayofweek, month, year, dayofyear, lag_value]],
        columns=["Hour", "DayOfWeek", "Month", "Year", "DayOfYear", "Lag_24h"],
    )

    prediction = model.predict(input_features)[0]
    st.success(f"🔮 Predicted Energy Consumption: **{prediction:,.2f} MW**")
