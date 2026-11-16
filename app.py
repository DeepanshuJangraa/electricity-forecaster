import datetime
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Electricity Consumption Forecaster", page_icon="⚡", layout="wide")

FEATURES = ["Hour", "DayOfWeek", "Month", "Year", "DayOfYear", "Lag_24h"]


import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()


def make_row(ts: pd.Timestamp, lag: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[ts.hour, ts.dayofweek, ts.month, ts.year, ts.dayofyear, lag]], columns=FEATURES
    )


def recursive_forecast(history: pd.Series, hours: int) -> pd.Series:
    """Predict hour by hour; each prediction feeds the next day's Lag_24h."""
    values = history.copy()
    preds = {}
    for _ in range(hours):
        ts = values.index[-1] + pd.Timedelta(hours=1)
        lag = values.loc[ts - pd.Timedelta(hours=24)]
        yhat = float(model.predict(make_row(ts, lag))[0])
        values.loc[ts] = yhat
        preds[ts] = yhat
    return pd.Series(preds, name="Forecast_MW")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚡ GridSight")
    st.caption("XGBoost hourly demand model")
    page = st.radio("View", ["Single prediction", "Forecast from history", "About the model"])

st.title("Electricity Consumption Forecaster")

# ---------------- Page 1: single prediction ----------------
if page == "Single prediction":
    st.write("Predict demand for one hour, given what was consumed at the same time yesterday.")
    c1, c2, c3 = st.columns(3)
    pred_date = c1.date_input("Date", datetime.date.today())
    pred_time = c2.time_input("Time", datetime.time(12, 0))
    lag_value = c3.number_input("Yesterday at this time (MW)", value=15000.0, step=100.0)

    if st.button("Predict consumption", type="primary"):
        ts = pd.Timestamp(datetime.datetime.combine(pred_date, pred_time))
        pred = float(model.predict(make_row(ts, lag_value))[0])
        delta = pred - lag_value
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted demand", f"{pred:,.0f} MW", f"{delta:+,.0f} MW vs yesterday")
        m2.metric("Day of week", ts.day_name())
        m3.metric("Hour", f"{ts.hour:02d}:00")
        if ts.year > 2018:
            st.warning(
                "The model was trained on data up to 2018. Because Year is a feature, "
                "dates far beyond that are extrapolations and may be less reliable."
            )

# ---------------- Page 2: forecast from history ----------------
elif page == "Forecast from history":
    st.write(
        "Upload recent hourly data (at least 24 hours, columns `Datetime` and `AEP_MW`). "
        "The app forecasts the next hours, using each prediction as the lag for the next day."
    )
    up = st.file_uploader("Hourly history CSV", type="csv")
    hours = st.slider("Hours to forecast", 24, 168, 72, step=24)

    if up is not None:
        df = pd.read_csv(up)
        if not {"Datetime", "AEP_MW"}.issubset(df.columns):
            st.error("The CSV needs two columns named Datetime and AEP_MW.")
            st.stop()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        hist = df.drop_duplicates("Datetime").set_index("Datetime").sort_index()["AEP_MW"]
        hist = hist.asfreq("h").interpolate()
        if len(hist) < 24:
            st.error("Need at least 24 hours of history.")
            st.stop()

        fc = recursive_forecast(hist, hours)
        recent = hist.iloc[-72:]

        fig = go.Figure()
        fig.add_scatter(x=recent.index, y=recent.values, name="History", line=dict(color="#e8a300"))
        fig.add_scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#0e7c86"))
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Megawatts (MW)", legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Peak forecast", f"{fc.max():,.0f} MW")
        k2.metric("Peak time", fc.idxmax().strftime("%a %H:%M"))
        k3.metric("Average forecast", f"{fc.mean():,.0f} MW")

        st.download_button(
            "Download forecast CSV",
            fc.rename_axis("Datetime").reset_index().to_csv(index=False),
            "forecast.csv", "text/csv",
        )
    else:
        st.info("Upload a CSV to see the forecast. A format example is the AEP_hourly.csv dataset.")

# ---------------- Page 3: about ----------------
else:
    st.subheader("How it works")
    st.write(
        "The model is an XGBoost regressor trained on the AEP hourly dataset. "
        "It sees six inputs and predicts demand in megawatts."
    )
    st.table(pd.DataFrame({
        "Feature": FEATURES,
        "Meaning": ["Hour of day (0 to 23)", "Monday = 0 to Sunday = 6", "Month (1 to 12)",
                    "Calendar year", "Day of year (1 to 366)", "Demand 24 hours earlier (MW)"],
    }))
    st.caption(
        f"Trees used: {getattr(model, 'best_iteration', 'n/a') + 1 if hasattr(model, 'best_iteration') else 'n/a'}. "
        "Trained on data before 2017, tested on 2017 to 2018."
    )
