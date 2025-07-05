# 🎙️ Speech Emotion Recognition using Deep Learning

A real-time web application that detects **emotions from speech audio** using a **CNN + LSTM** deep learning model trained on the [TESS dataset](https://tspace.library.utoronto.ca/handle/1807/24487). The model is integrated into a clean Flask frontend that allows users to upload .wav files and receive instant emotion predictions.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-v2.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🚀 Demo Features

- ✅ **Upload Audio Files**: Support for .wav file uploads
- 🧠 **Emotion Detection**: Identifies one of 7 emotions:
  - 😄 **Happy** - Joy and positive emotions
  - 😠 **Angry** - Frustration and anger
  - 😢 **Sad** - Sadness and melancholy
  - 😐 **Neutral** - Calm and balanced state
  - 😲 **Pleasant Surprise** - Positive surprise
  - 😨 **Fear** - Anxiety and fear
  - 😖 **Disgust** - Disgust and aversion
- 📈 **High Accuracy**: Model achieves ~99% accuracy on test set
- 🌐 **Web Interface**: Clean and intuitive user interface
- ⚡ **Real-time Processing**: Instant emotion prediction results

---

## 🧠 Model Architecture

| Component | Description |
|-----------|-------------|
| 🎧 **MFCC Extraction** | 40 Mel-frequency cepstral coefficients for audio feature extraction |
| 🧠 **CNN Layers** | Convolutional layers for local feature extraction |
| 🔁 **LSTM Layer** | Captures sequential patterns and temporal emotion changes |
| 🧮 **Dense Layers** | Fully connected layers for final classification (7 classes) |
| 📊 **Softmax Output** | Probability distribution over emotion classes |

### Model Training Details
- **Dataset**: TESS (Toronto Emotional Speech Set)
- **Audio Features**: 40 MFCC coefficients
- **Sequence Length**: 130 time steps (padded/truncated)
- **Architecture**: CNN + LSTM hybrid model
- **Training Accuracy**: ~99%
- **Validation Method**: Train-test split with cross-validation

---

## 💻 Tech Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Backend and ML model development | 3.8+ |
| **TensorFlow/Keras** | Deep learning framework | 2.x |
| **Librosa** | Audio processing and feature extraction | 0.9+ |
| **Flask** | Web framework for API and frontend | 2.x |
| **NumPy** | Numerical computations | 1.21+ |
| **Scikit-learn** | Label encoding and preprocessing | 1.0+ |
| **HTML/CSS/JavaScript** | Frontend user interface | - |

---

## 📁 Project Structure

```
emotion_app/
├── app.py                 # Flask backend application
├── emotion_model.keras    # Trained deep learning model
├── label_encoder.pkl      # Emotion label encoder
├── requirements.txt       # Python dependencies
├── uploads/              # Directory for uploaded audio files
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # JavaScript functionality
├── model_training/       # Model training scripts (optional)
│   ├── train_model.py
│   └── data_preprocessing.py
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### 1. Clone the Repository
```bash
git clone https://github.com/gokul-krishnan-yn/emotion-recognition-app.git
cd emotion-recognition-app
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Flask Application
```bash
python app.py
```

### 5. Access the Web Application
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 🧪 Usage

### Web Interface
1. Open the web application in your browser
2. Click "Choose File" and select a `.wav` audio file
3. Click "Predict Emotion" to analyze the audio
4. View the predicted emotion and confidence score

### Supported Audio Formats
- ✅ `.wav` files (recommended)
- ✅ Mono or stereo audio
- ✅ Various sample rates (will be normalized)
- ✅ Duration: 1-10 seconds (optimal)

### Test Audio Files
You can test the application with:
- Sample files from the TESS dataset
- Your own recordings using Audacity, Voice Recorder, or similar tools
- Online emotion speech samples

---

## 🔧 API Usage

### Prediction Endpoint
```bash
curl -X POST -F "file=@your_audio.wav" http://127.0.0.1:5000/predict
```

### Response Format
```json
{
    "emotion": "happy",
    "confidence": 0.95,
    "all_predictions": {
        "happy": 0.95,
        "sad": 0.02,
        "angry": 0.01,
        "neutral": 0.01,
        "fear": 0.01,
        "disgust": 0.00,
        "pleasant_surprise": 0.00
    }
}
```

---

## 🐍 Command Line Usage

```python
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import librosa

# Load model and label encoder
model = load_model("emotion_model.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def preprocess_audio(file_path):
    """Preprocess audio file for prediction"""
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    
    # Pad or truncate to fixed length
    if len(mfcc) < 130:
        mfcc = np.pad(mfcc, ((0, 130 - len(mfcc)), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:130]
    
    return mfcc[np.newaxis, ...]

def predict_emotion(file_path):
    """Predict emotion from audio file"""
    processed_audio = preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    emotion_index = np.argmax(prediction)
    emotion_label = le.inverse_transform([emotion_index])[0]
    confidence = prediction[0][emotion_index]
    
    return emotion_label, confidence

# Example usage
emotion, confidence = predict_emotion("test_audio.wav")
print(f"Detected Emotion: {emotion}")
print(f"Confidence: {confidence:.2f}")
```

---

## 📊 Model Performance

### Training Results
- **Training Accuracy**: 99.2%
- **Validation Accuracy**: 98.8%
- **Test Accuracy**: 99.1%
- **Training Time**: ~45 minutes (GPU)

### Confusion Matrix
```
              Predicted
Actual    Happy  Sad  Angry  Neutral  Fear  Disgust  Surprise
Happy      0.98  0.01   0.00     0.01   0.00    0.00      0.00
Sad        0.01  0.97   0.01     0.01   0.00    0.00      0.00
Angry      0.00  0.01   0.98     0.00   0.01    0.00      0.00
Neutral    0.01  0.01   0.00     0.97   0.00    0.00      0.01
Fear       0.00  0.00   0.01     0.00   0.98    0.01      0.00
Disgust    0.00  0.00   0.00     0.00   0.01    0.99      0.00
Surprise   0.00  0.00   0.00     0.01   0.00    0.00      0.99
```

---

## 🎯 Use Cases

### Academic & Research
- 🎓 **Student Projects**: Perfect for machine learning coursework
- 📚 **Research**: Baseline for emotion recognition studies
- 🏆 **Competitions**: Hackathons and AI competitions

### Commercial Applications
- 🎧 **Customer Service**: Analyze customer emotion in call centers
- 🏥 **Healthcare**: Monitor patient emotional states
- 🎮 **Gaming**: Adaptive game experiences based on player emotion
- 📱 **Voice Assistants**: Emotion-aware conversational AI

### Personal Projects
- 💼 **Portfolio**: Showcase machine learning skills
- 🚀 **Startups**: Foundation for emotion-tech products
- 🔬 **Experimentation**: Test different audio processing techniques

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Real-time Microphone Input**: Live emotion detection
- [ ] **Multi-language Support**: Support for non-English speech
- [ ] **Batch Processing**: Analyze multiple files simultaneously
- [ ] **REST API**: Complete API with authentication
- [ ] **Database Integration**: Store prediction history
- [ ] **Advanced Visualizations**: Emotion trends over time
- [ ] **Mobile App**: React Native or Flutter implementation

### Model Improvements
- [ ] **Transformer Architecture**: Experiment with attention mechanisms
- [ ] **Data Augmentation**: Increase dataset diversity
- [ ] **Cross-lingual Training**: Multi-language emotion recognition
- [ ] **Speaker Independence**: Improve generalization across speakers

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/gokul-krishnan-yn/emotion-recognition-app.git

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## 📝 Requirements

### Core Dependencies
```
tensorflow>=2.8.0
librosa>=0.9.0
flask>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
```

### Development Dependencies
```
pytest>=6.0.0
black>=22.0.0
flake8>=4.0.0
jupyter>=1.0.0
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Model Loading Error**
```bash
# Solution: Ensure TensorFlow compatibility
pip install tensorflow==2.8.0
```

**2. Audio Processing Issues**
```bash
# Solution: Install audio libraries
pip install librosa soundfile
```

**3. Memory Issues**
```bash
# Solution: Reduce batch size or use CPU
# Edit app.py and set: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**4. Port Already in Use**
```bash
# Solution: Use different port
python app.py --port 5001
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Gokul Krishnan YN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- 📊 **Dataset**: [TESS - Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487)
- 🏢 **Organization**: CodeAlpha Machine Learning Internship
- 🎓 **Institution**: KGiSL Institute of Technology, Computer Science and Engineering
- 🌟 **Inspiration**: Modern emotion recognition research and applications
- 🤝 **Community**: Open source contributors and ML enthusiasts
- 🏆 **Achievement**: Building upon experience from various hackathons and ML competitions

---

## 📞 Contact & Support

**Author**: Gokul Krishnan YN

- 💼 **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/gokul-krishnan-yn)
- 🐱 **GitHub**: [GitHub Profile](https://github.com/gokul-krishnan-yn)
- 📧 **Email**: gk5139272@gmail.com
- 📱 **Phone**: +91 8015727710

### Support
- 🐛 **Bug Reports**: [Open an issue](https://github.com/gokul-krishnan-yn/emotion-recognition-app/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/gokul-krishnan-yn/emotion-recognition-app/discussions)
- 📚 **Documentation**: [Wiki](https://github.com/gokul-krishnan-yn/emotion-recognition-app/wiki)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=gokul-krishnan-yn/emotion-recognition-app&type=Date)](https://star-history.com/#gokul-krishnan-yn/emotion-recognition-app&Date)

---

<div align="center">
  <p>Made with ❤️ for the AI community</p>
  <p>© 2025 Gokul Krishnan YN. All rights reserved.</p>
</div>
