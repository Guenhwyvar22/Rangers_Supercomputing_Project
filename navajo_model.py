#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def extract_features(file_path, duration=3):
    y, sr = librosa.load(file_path, duration=duration, sr=None)
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    target_shape = (n_mels, int(sr * duration / 512) + 1)

    mel_spec_resized = librosa.util.fix_length(mel_spec_db, target_shape[1] * 512, axis=1, mode='constant', constant_values=0.0)

    return mel_spec_resized


data_dir = '/home/ljb/Documents/'
metadata = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

X = np.array([extract_features(os.path.join(data_dir, file), duration=3) for file in metadata['file_path']])

data_dir = "/home/ljb/Documents/"
metadata = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

X = np.array([extract_features(os.path.join(data_dir, file), duration=3) for file in metadata['file_path']])

label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(metadata['label']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
