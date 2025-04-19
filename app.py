from flask import Flask, request, render_template, jsonify
from utils.audio_processing import process_audio
import torch
import numpy as np

app = Flask(__name__)

try:
    model = torch.load("heart_model.pt", map_location=torch.device("cpu"))
    model.eval()
    use_model = True
except:
    print("Model not found. Using dummy mode.")
    use_model = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files['audio']
    features = process_audio(audio_file)
    if use_model:
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32)
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            result = "Normal" if prediction == 0 else "Abnormal"
    else:
        result = "Normal" if np.random.rand() > 0.5 else "Abnormal"
    return jsonify({"heart_status": result})

if __name__ == '__main__':
    app.run(debug=True)
