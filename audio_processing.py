import librosa
import numpy as np
import soundfile as sf
import tempfile

def process_audio(uploaded_file):
    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Now read it from disk
    y, sr = sf.read(tmp_path)
    y = y[:sr * 5]  # first 5 seconds only
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.expand_dims(mfccs, axis=0)
