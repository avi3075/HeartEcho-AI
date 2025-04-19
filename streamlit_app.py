import streamlit as st
import torch
import numpy as np
from utils.audio_processing import process_audio

st.title("HeartEcho AI - Streamlit Version")

try:
    model = torch.load("heart_model.pt", map_location=torch.device("cpu"))
    model.eval()
    use_model = True
except:
    st.warning("Model not found. Running in dummy mode.")
    use_model = False

uploaded_file = st.file_uploader("Upload a heart sound (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    features = process_audio(uploaded_file)
    if use_model:
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32)
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            result = "Normal" if prediction == 0 else "Abnormal"
    else:
        result = "Normal" if np.random.rand() > 0.5 else "Abnormal"
    st.success(f"Result: {result}")
