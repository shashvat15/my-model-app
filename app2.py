
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, time
import pytz

# Add other necessary imports as you build the app

@st.cache_resource
def load_improved_models():
    """Loads the trained improved XGBoost and TensorFlow Lite models."""
    try:
        # Load improved XGBoost models for sin(azimuth), cos(azimuth), and elevation
        xgb_azimuth_sin_model = joblib.load("xgb_azimuth_sin_improved.pkl")
        xgb_azimuth_cos_model = joblib.load("xgb_azimuth_cos_improved.pkl")
        xgb_elevation_model = joblib.load("xgb_elevation_improved.pkl")

        # Load improved TFLite models
        interpreter_az_sin = tf.lite.Interpreter(model_path="sunpos_az_sin_model_improved.tflite")
        interpreter_az_sin.allocate_tensors()

        interpreter_az_cos = tf.lite.Interpreter(model_path="sunpos_az_cos_model_improved.tflite")
        interpreter_az_cos.allocate_tensors()

        interpreter_el = tf.lite.Interpreter(model_path="sunpos_el_model_improved.tflite")
        interpreter_el.allocate_tensors()

        return (xgb_azimuth_sin_model, xgb_azimuth_cos_model, xgb_elevation_model,
                interpreter_az_sin, interpreter_az_cos, interpreter_el)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Display the title first
st.title("Improved Sun Position Predictor (XGBoost vs MLP)")

# Load the models after Streamlit has initialized
models = load_improved_models()

if models is None:
    st.error("Failed to load models. Please check that all model files are present.")
    st.stop()

# Unpack the models
(xgb_azimuth_sin_model, xgb_azimuth_cos_model, xgb_elevation_model,
 tflite_interpreter_az_sin, tflite_interpreter_az_cos, tflite_interpreter_el) = models

# ============================
# STEP 1: Input Widgets
# ============================

st.header("Input Parameters")

# Date and Time inputs
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Select Date", datetime.now().date())
with col2:
    input_time = st.time_input("Select Time", time(12, 0))  # Fixed: Use a fixed default time instead of current time

# Location inputs
col3, col4 = st.columns(2)
with col3:
    input_latitude = st.number_input("Enter Latitude", value=13.0827, format="%.4f")
with col4:
    input_longitude = st.number_input("Enter Longitude", value=80.2707, format="%.4f")

# Predict button
predict_button = st.button("Predict Sun Position")

# ============================
# STEP 2: Prediction Logic
# ============================

def predict_sun_position(date, time, latitude, longitude,
                         xgb_az_sin_model, xgb_az_cos_model, xgb_el_model,
                         tflite_interp_az_sin, tflite_interp_az_cos, tflite_interp_el):
    """Predicts sun azimuth and elevation using trained models."""
    # Combine date and time
    dt = datetime.combine(date, time)

    # Add time and day of year features
    df_input = pd.DataFrame({
        "year": [dt.year],
        "month": [dt.month],
        "day": [dt.day],
        "hour": [dt.hour],
        "minute": [dt.minute],
        "latitude": [latitude],
        "longitude": [longitude]
    })

    df_input["hour_frac"] = df_input["hour"] + df_input["minute"]/60.0
    df_input["hour_sin"] = np.sin(2*np.pi*df_input["hour_frac"]/24)
    df_input["hour_cos"] = np.cos(2*np.pi*df_input["hour_frac"]/24)
    df_input["doy"] = pd.to_datetime(dict(year=df_input.year, month=df_input.month, day=df_input.day)).dt.dayofyear
    df_input["doy_sin"] = np.sin(2*np.pi*df_input["doy"]/365)
    df_input["doy_cos"] = np.cos(2*np.pi*df_input["doy"]/365)

    features = ["hour_sin","hour_cos","doy_sin","doy_cos","latitude","longitude"]
    X_input = df_input[features]

    # Predict using XGBoost
    xgb_azimuth_sin_pred = xgb_az_sin_model.predict(X_input)[0]
    xgb_azimuth_cos_pred = xgb_az_cos_model.predict(X_input)[0]
    xgb_elevation_pred = xgb_el_model.predict(X_input)[0]

    # Recover XGBoost Azimuth
    xgb_azimuth_pred = np.rad2deg(np.arctan2(xgb_azimuth_sin_pred, xgb_azimuth_cos_pred))
    xgb_azimuth_pred = (xgb_azimuth_pred + 360) % 360


    # Predict using TFLite
    input_details_az_sin = tflite_interp_az_sin.get_input_details()
    output_details_az_sin = tflite_interp_az_sin.get_output_details()
    tflite_input_data = X_input.astype(np.float32).values
    tflite_interp_az_sin.set_tensor(input_details_az_sin[0]['index'], tflite_input_data)
    tflite_interp_az_sin.invoke()
    tflite_azimuth_sin_pred = tflite_interp_az_sin.get_tensor(output_details_az_sin[0]['index'])[0][0]

    input_details_az_cos = tflite_interp_az_cos.get_input_details()
    output_details_az_cos = tflite_interp_az_cos.get_output_details()
    tflite_interp_az_cos.set_tensor(input_details_az_cos[0]['index'], tflite_input_data)
    tflite_interp_az_cos.invoke()
    tflite_azimuth_cos_pred = tflite_interp_az_cos.get_tensor(output_details_az_cos[0]['index'])[0][0]

    input_details_el = tflite_interp_el.get_input_details()
    output_details_el = tflite_interp_el.get_output_details()
    tflite_interp_el.set_tensor(input_details_el[0]['index'], tflite_input_data)
    tflite_interp_el.invoke()
    tflite_elevation_pred = tflite_interp_el.get_tensor(output_details_el[0]['index'])[0][0]


    # Recover TFLite Azimuth
    tflite_azimuth_pred = np.rad2deg(np.arctan2(tflite_azimuth_sin_pred, tflite_azimuth_cos_pred))
    tflite_azimuth_pred = (tflite_azimuth_pred + 360) % 360


    return xgb_azimuth_pred, xgb_elevation_pred, tflite_azimuth_pred, tflite_elevation_pred

# ============================
# STEP 3: Display Predictions
# ============================
if predict_button:
    xgb_az, xgb_el, tflite_az, tflite_el = predict_sun_position(
        input_date, input_time, input_latitude, input_longitude,
        xgb_azimuth_sin_model, xgb_azimuth_cos_model, xgb_elevation_model,
        tflite_interpreter_az_sin, tflite_interpreter_az_cos, tflite_interpreter_el
    )

    st.subheader("Predicted Sun Position")

    col_xgb, col_mlp = st.columns(2)

    with col_xgb:
        st.write("**XGBoost Predictions:**")
        st.write(f"Azimuth: {xgb_az:.2f} degrees")
        st.write(f"Elevation: {xgb_el:.2f} degrees")

    with col_mlp:
        st.write("**TensorFlow Lite (MLP) Predictions:**")
        st.write(f"Azimuth: {tflite_az:.2f} degrees")
        st.write(f"Elevation: {tflite_el:.2f} degrees")