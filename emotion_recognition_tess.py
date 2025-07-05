import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# ========= CONFIG =========
DATA_PATH = "Data set"  # ✅ CHANGE THIS TO YOUR TESS FOLDER PATH
MAX_LEN = 130  # Max time steps for padding

EMOTION_MAP = {
    'ps': 'pleasant_surprise'
}

# ========= FEATURE EXTRACTION =========
def extract_sequence_features(path, max_len=130):
    features = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                emotion = file.split('_')[-1].replace('.wav', '').lower()
                emotion = EMOTION_MAP.get(emotion, emotion)
                file_path = os.path.join(root, file)

                try:
                    y, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T

                    if len(mfcc) > max_len:
                        mfcc = mfcc[:max_len]
                    else:
                        pad_width = max_len - len(mfcc)
                        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

                    features.append([mfcc, emotion])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return pd.DataFrame(features, columns=['feature', 'label'])

# ========= LOAD DATA =========
print("🔍 Extracting features...")
df = extract_sequence_features(DATA_PATH, max_len=MAX_LEN)
print(f"✅ Loaded {len(df)} samples.")

# ========= ENCODE LABELS =========
X = np.array(df['feature'].tolist())  # shape: (samples, time_steps, 40)
le = LabelEncoder()
y = le.fit_transform(df['label'])
y = to_categorical(y)

# Save Label Encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ========= SPLIT DATA =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========= BUILD MODEL =========
print("🔧 Building CNN + LSTM model...")
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(MAX_LEN, 40)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========= TRAIN MODEL =========
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

# ========= EVALUATE =========
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📄 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# ========= PLOT ACCURACY =========
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("CNN + LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========= SAVE MODEL =========
print("💾 Saving model...")
model.save("emotion_model.keras")   # ✅ Recommended Keras format
model.save("emotion_model.h5")      # ✅ Legacy-compatible HDF5 format
print("✅ Model and label encoder saved successfully!")
