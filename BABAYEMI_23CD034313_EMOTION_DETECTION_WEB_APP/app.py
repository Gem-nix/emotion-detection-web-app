# app.py - THE BACKEND OF YOUR WEB APP
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sqlite3

# === SETUP ===
app = Flask(__name__)
app.config['UPLOADS'] = 'static/uploads'
os.makedirs(app.config['UPLOADS'], exist_ok=True)

# === LOAD YOUR RETRAINED MODEL ===
model = load_model('AYOMI_EMOTION_AI_2025.h5')  # ← YOUR MODEL!
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === DATABASE ===
def init_db():
    conn = sqlite3.connect('emotion_records.db')
    conn.execute('CREATE TABLE IF NOT EXISTS records (name TEXT, emotion TEXT, image TEXT)')
    conn.close()
init_db()

# === PREDICT EMOTION ===
def predict_emotion(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        pred = model.predict(roi, verbose=0)
        emotion = EMOTIONS[np.argmax(pred)]
        
        # Draw box and label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return emotion, img
    return None, None

# === ROUTE ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        emotion, result_img = predict_emotion(img.copy())
        if emotion:
            path = f"static/uploads/{name}_{emotion}.jpg"
            cv2.imwrite(path, result_img)
            
            # Save to database
            conn = sqlite3.connect('emotion_records.db')
            conn.execute('INSERT INTO records VALUES (?, ?, ?)', (name, emotion, path))
            conn.commit()
            conn.close()
            
            return render_template('index.html', name=name, emotion=emotion, image=path)
        else:
            return render_template('index.html', error="No face detected!")
    
    return render_template('index.html')

# === RUN ===
if __name__ == '__main__':
    app.run(debug=True)
