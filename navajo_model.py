#!/usr/bin/env python3
#
#!/usr/bin/env python3
#
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Define the extract_features function
def extract_features(file_path, duration=3):
    y, sr = librosa.load(file_path, duration=duration, sr=None)
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Adjust the target shape based on the mel spectrogram shape
    target_shape = (n_mels, int(sr * duration / 512) + 1)

    # Fix the length using the target shape
    mel_spec_resized = librosa.util.fix_length(mel_spec_db, target_shape[1], axis=1, mode='constant', constant_values=0.0)

    return mel_spec_resized


# Read metadata from the dataset.csv file
data_dir = '/home/layla/Documents/'
metadata = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

# Extract features for each file in the metadata
X = np.array([extract_features(os.path.join(data_dir, file), duration=3) for file in metadata['file_path']])

# Rest of your code...
# Add the rest of your code here using the extracted features stored in the variable 'X'

# Print warning about Pyarrow deprecation
print("Pyarrow deprecation warning:")
print("Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),")
print("but was not found to be installed on your system.")
print("If this would cause problems for you, please provide feedback at https://github.com/pandas-dev/pandas/issues/54466")
print()

# Load data using pandas
data_dir = "/home/layla/Documents/"
metadata = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

# Extract features for each audio file
X = np.array([extract_features(os.path.join(data_dir, file), duration=3) for file in metadata['file_path']])

# Encode the labels
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(metadata['label']))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
