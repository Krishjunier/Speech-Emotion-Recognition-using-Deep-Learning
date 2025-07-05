from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and label encoder
model = load_model("emotion_model.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def preprocess_audio(file_path, max_len=130):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    if len(mfcc) > max_len:
        mfcc = mfcc[:max_len]
    else:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)), mode='constant')
    return mfcc[np.newaxis, ...]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['audio']
        if not file:
            return render_template('index.html', error="No audio file provided")
        
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process audio
        X_input = preprocess_audio(file_path)
        pred = model.predict(X_input)
        emotion = le.inverse_transform([np.argmax(pred)])[0]

        # Clean up file
        os.remove(file_path)
        
        return render_template('index.html', prediction=emotion)
    
    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return render_template('index.html', error=f"Error processing audio: {str(e)}")

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)