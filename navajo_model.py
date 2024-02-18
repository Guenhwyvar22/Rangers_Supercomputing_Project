#!/usr/bin/env python3

from pydub import AudioSegment
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import librosa

# Function to extract features (spectrogram) from audio file using pydub and librosa
def extract_features(file_path, duration=3, sr=44100, n_mels=128):
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    y = np.array(audio.get_array_of_samples())

    # Correct the dimension of the input array for librosa
    y = y.reshape(-1)

    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.reshape((n_mels, -1, 1))

# Rest of your code remains the same
