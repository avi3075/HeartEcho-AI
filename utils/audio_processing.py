import librosa
import numpy as np
import soundfile as sf

def process_audio(file):
    y, sr = sf.read(file)
    y = y[:sr*5]  # first 5 seconds
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.expand_dims(mfccs, axis=0)  # 2D input
