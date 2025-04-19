import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

def process_audio(file):
    if hasattr(file, 'read'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
    else:
        tmp_path = file

    try:
        y, sr = sf.read(tmp_path)
        y = y[:sr * 5]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.expand_dims(mfccs, axis=0)
    finally:
        if hasattr(file, 'read'):
            os.unlink(tmp_path)
