import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import torch
import tempfile
import soundfile as sf
from utils.audio_processing import process_audio

st.title("🎙️ HeartEcho AI – Mic + Upload Analysis")

# Load model
try:
    model = torch.load("heart_model.pt", map_location=torch.device("cpu"))
    model.eval()
    use_model = True
except:
    st.warning("Model not found. Running in dummy mode.")
    use_model = False

st.header("🔘 Option 1: Upload Heart Sound (.wav)")
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
    st.success(f"🩺 Result: {result}")

st.markdown("---")
st.header("🎙️ Option 2: Record Using Microphone")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = b""

    def recv(self, frame):
        self.buffer += frame.to_ndarray().tobytes()
        return frame

ctx = webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor:
    if st.button("Analyze Mic Recording"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(ctx.audio_processor.buffer)
            f.seek(0)
            try:
                y, sr = sf.read(f.name)
                if len(y) > sr * 5:
                    y = y[:sr * 5]

                features = process_audio(f.name)

                if use_model:
                    with torch.no_grad():
                        input_tensor = torch.tensor(features, dtype=torch.float32)
                        output = model(input_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                        result = "Normal" if prediction == 0 else "Abnormal"
                else:
                    result = "Normal" if np.random.rand() > 0.5 else "Abnormal"

                st.success(f"🩺 Result: {result}")
            except Exception as e:
                st.error("Recording failed. Try again.")
