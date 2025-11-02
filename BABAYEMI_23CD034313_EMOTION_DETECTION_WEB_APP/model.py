# model.py - FULL TRAINING SCRIPT (RUN IN GOOGLE COLAB)
# This is the EXACT code used to retrain the model

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load data
data = pd.read_csv('fer2013.csv')

def preprocess(df):
    imgs = []
    labels = []
    for _, row in df.iterrows():
        img = np.fromstring(row['pixels'], sep=' ', dtype=int).reshape(48, 48, 1)
        imgs.append(img)
        labels.append(row['emotion'])
    return np.array(imgs), np.array(labels)

train = data[data['Usage'] == 'Training']
test = data[data['Usage'] == 'PublicTest']

X_train, y_train = preprocess(train)
X_test, y_test = preprocess(test)

X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save('AYOMI_EMOTION_AI_2025.h5')
print("Model saved as AYOMI_EMOTION_AI_2025.h5")
