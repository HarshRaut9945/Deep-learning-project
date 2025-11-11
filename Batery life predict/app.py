import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow's Keras
import pickle
import os

# Load TensorFlow model

 #Define absolute paths for model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "battery_life_model.h5")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
label_path=os.path.join(base_dir, "models", "label_encoder.pkl")


model = load_model(model_path)
scaler=pickle.load(open(scaler_path,'rb'))
label_encoder=pickle.load(open(label_path,'rb'))


# Prediction function
def predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model):
    # Encode the categorical feature
    type_discharge_encoded = label_encoder.transform([type_discharge])[0]

    # Prepare the input feature vector
    X_input = np.array([[type_discharge_encoded,Capacity, Re, Rct]])

    # Scale the input features using the same scaler
    X_input_scaled = scaler.transform(X_input)

    # Predict the battery life (ambient_temperature)
    predicted_battery_life = model.predict(X_input_scaled)

    return predicted_battery_life[0]



# Streamlit app UI
st.title("Battery Life Prediction using ANN")

# User input fields
type_discharge = st.selectbox("Select Discharge Type", ['charge', 'discharge', 'impedance'])
Capacity = st.number_input("Enter Capacity", min_value=0.0)
Re = st.number_input("Enter Re", min_value=-1e12, max_value=1e12)
Rct = st.number_input("Enter Rct", min_value=-1e12, max_value=1e12)

# Button to make prediction
if st.button('Predict Battery Life'):
    predicted_battery_life = predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model)
    st.write(f"The predicted battery life is: {predicted_battery_life} units")