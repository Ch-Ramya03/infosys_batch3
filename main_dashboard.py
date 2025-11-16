# ==============================================================
# 🌍 Air Quality Forecast Dashboard (Full Milestone Version)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# --------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

cities = df['City'].dropna().unique()
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

# Sidebar Navigation
st.sidebar.title("📁 Navigation")
section = st.sidebar.radio(
    "Go to:",
    [
        "🏁 Milestone 1: Data Preprocessing & EDA",
        "📊 Milestone 2: Model Training & Evaluation (ARIMA, Prophet, LSTM)",
        "🚨 Milestone 3: Alert Logic & Trend Visualization",
        "🌐 Milestone 4: Interactive Forecast Dashboard"
    ]
)

# ==============================================================
# 🏁 Milestone 1 – Data Preprocessing & EDA
# ==============================================================
if "Data Preprocessing" in section:
    st.header("🏁 Milestone 1: Data Preprocessing & EDA")
    st.write("### Step 1: Raw Dataset Preview")
    st.dataframe(df.head())

    # Data Cleaning
    st.write("### Step 2: Cleaning Missing Values & Outliers")
    cleaned_df = df.copy()
    cleaned_df[pollutants] = cleaned_df[pollutants].fillna(method='ffill').fillna(method='bfill')
    cleaned_df = cleaned_df.dropna(subset=['City', 'Date'])

    st.success(f"✅ Cleaned dataset created with {len(cleaned_df)} rows.")

    # Download cleaned dataset
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned Dataset", csv, "cleaned_city_day.csv", "text/csv")

    st.write("### Step 3: Missing Values Summary")
    st.dataframe(df.isnull().sum())

    st.write("### Step 4: Correlation Heatmap of Pollutants")
    st.plotly_chart(px.imshow(df[pollutants].corr(), text_auto=True, title="Pollutant Correlations"), use_container_width=True)

# ==============================================================
# 📊 Milestone 2 – Model Training & Evaluation
# ==============================================================
elif "Model Training" in section:
    st.header("📊 Milestone 2: Model Training & Evaluation")

    city = st.selectbox("Select City", cities)
    pollutant = st.selectbox("Select Pollutant", pollutants)
    model_type = st.radio("Choose Model:", ["ARIMA", "Prophet", "LSTM"])

    city_df = df[df['City'] == city].sort_values('Date')
    data = city_df[['Date', pollutant]].dropna()
    data = data.set_index('Date')

    if len(data) < 60:
        st.warning("⚠️ Not enough data points for training.")
        st.stop()

    train = data[:-30]
    test = data[-30:]

    # =============== ARIMA MODEL ===============
    if model_type == "ARIMA":
        model = ARIMA(train, order=(3, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

    # =============== PROPHET MODEL ===============
    elif model_type == "Prophet":
        df_prophet = train.reset_index().rename(columns={'Date': 'ds', pollutant: 'y'})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future).set_index('ds').iloc[-30:]['yhat']

    # =============== LSTM MODEL ===============
    elif model_type == "LSTM":
        series = train.values
        generator = TimeseriesGenerator(series, series, length=10, batch_size=8)
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(10, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=20, verbose=0)
        forecast = []
        current_batch = series[-10:].reshape((1, 10, 1))
        for i in range(30):
            pred = model.predict(current_batch)[0]
            forecast.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        forecast = pd.Series([x[0] for x in forecast], index=test.index)

    # ================== Evaluation ==================
    mae = mean_absolute_error(test[pollutant], forecast)
    rmse = math.sqrt(mean_squared_error(test[pollutant], forecast))
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    result_df = pd.concat([train, test], axis=0)
    result_df['Forecast'] = np.nan
    result_df.loc[test.index, 'Forecast'] = forecast

    st.plotly_chart(
        px.line(result_df, y=[pollutant, 'Forecast'], title=f"{model_type} Forecast for {pollutant} in {city}"),
        use_container_width=True
    )

# ==============================================================
# 🚨 Milestone 3 – Alert Logic & Visualization
# ==============================================================
elif "Alert Logic" in section:
    st.header("🚨 Milestone 3: Alert Logic & Trend Visualization")

    city = st.selectbox("Select City", cities)
    pollutant = st.selectbox("Select Pollutant for Trend", pollutants)
    city_df = df[df['City'] == city].sort_values('Date')

    def classify_aqi(value):
        if value <= 50: return "Good"
        elif value <= 100: return "Moderate"
        elif value <= 200: return "Poor"
        elif value <= 300: return "Very Poor"
        else: return "Severe"

    city_df['AQI_Category'] = city_df[pollutant].apply(classify_aqi)
    city_df['Color'] = city_df['AQI_Category'].map({
        "Good": "green", "Moderate": "yellow", "Poor": "orange",
        "Very Poor": "red", "Severe": "purple"
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df['Date'], y=city_df[pollutant],
        mode='lines+markers', marker=dict(color=city_df['Color']),
        name=f"{pollutant} AQI"
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(city_df[['Date', pollutant, 'AQI_Category']].tail(10))

# ==============================================================
# 🌐 Milestone 4 – Interactive Dashboard
# ==============================================================
elif "Interactive" in section:
    st.header("🌐 Milestone 4: Interactive Forecast Dashboard")

    city = st.selectbox("Select City", cities)
    pollutant = st.selectbox("Select Pollutant", pollutants)

    city_df = df[df['City'] == city].sort_values('Date')
    data = city_df[['Date', pollutant]].dropna()

    st.write("### Historical Trend")
    st.plotly_chart(px.line(data, x='Date', y=pollutant, title=f"{pollutant} Trend – {city}"), use_container_width=True)

    st.write("### Simple 7-Day Forecast (ARIMA)")
    try:
        model = ARIMA(data.set_index('Date'), order=(3, 1, 2))
        model_fit = model.fit()
        future_forecast = model_fit.forecast(steps=7)
        forecast_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, 8)]
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_forecast})
        st.line_chart(forecast_df.set_index('Date'))
    except:
        st.warning("Forecast not available for this pollutant.")
