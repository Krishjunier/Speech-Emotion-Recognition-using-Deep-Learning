let mediaRecorder;
let recordedChunks = [];
let recordedBlob;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const audioPreview = document.getElementById('audioPreview');
const audioPlayer = document.getElementById('audioPlayer');
const submitBtn = document.getElementById('submitBtn');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const resultSection = document.getElementById('resultSection');
const resultEmotion = document.getElementById('resultEmotion');
const resultDescription = document.getElementById('resultDescription');
const emotionForm = document.getElementById('emotionForm');

// Emotion descriptions
const emotionDescriptions = {
    'angry': 'Strong displeasure or hostility detected in your voice 😠',
    'disgust': 'A feeling of revulsion or strong disapproval 🤢',
    'fear': 'Anxiety or apprehension detected 😨',
    'happy': 'Joy and positivity shine through your voice! 😊',
    'neutral': 'Calm and balanced emotional state 😐',
    'sad': 'Melancholy or sorrow detected in your tone 😢',
    'surprise': 'Unexpected emotion or astonishment 😲'
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Set description for existing prediction (if any)
    if (resultEmotion && resultDescription) {
        const emotion = resultEmotion.textContent.toLowerCase().trim();
        resultDescription.textContent = emotionDescriptions[emotion] || 'Emotion detected in your voice';
        resultSection.classList.add('show');
    }
});

// File upload handling
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        fileName.textContent = file.name;
        fileInfo.classList.add('show');
        submitBtn.disabled = false;
        hideError();
        hideResult();
        
        // Hide audio preview if showing recorded audio
        audioPreview.style.display = 'none';
        recordedBlob = null;
    }
}

// Audio recording
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(recordedBlob);
            audioPlayer.src = audioUrl;
            audioPreview.style.display = 'block';
            submitBtn.disabled = false;
            
            // Clear file input
            fileInput.value = '';
            fileInfo.classList.remove('show');
            
            hideError();
            hideResult();
        };

        mediaRecorder.start();
        recordBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        recordBtn.classList.add('recording');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showError('Unable to access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    recordBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    recordBtn.classList.remove('recording');
}

// Form submission
emotionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    
    if (recordedBlob) {
        formData.append('audio', recordedBlob, 'recorded_audio.wav');
    } else if (fileInput.files[0]) {
        formData.append('audio', fileInput.files[0]);
    } else {
        showError('Please select an audio file or record your voice.');
        return;
    }

    showLoading();
    hideError();
    hideResult();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.text();
        
        // Parse the emotion from the response
        const parser = new DOMParser();
        const doc = parser.parseFromString(result, 'text/html');
        const emotionElement = doc.querySelector('#resultEmotion');
        
        if (emotionElement) {
            const emotion = emotionElement.textContent.trim();
            showResult(emotion);
        } else {
            // Fallback: reload the page to show the result
            window.location.reload();
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Error analyzing audio. Please try again.');
    } finally {
        hideLoading();
    }
});

// Utility functions
function showLoading() {
    loading.classList.add('show');
    submitBtn.disabled = true;
}

function hideLoading() {
    loading.classList.remove('show');
    submitBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

function hideError() {
    errorMessage.classList.remove('show');
}

function showResult(emotion) {
    resultEmotion.textContent = emotion;
    resultDescription.textContent = emotionDescriptions[emotion.toLowerCase()] || 'Emotion detected in your voice';
    resultSection.classList.add('show');
}

function hideResult() {
    resultSection.classList.remove('show');
}

// Clean up audio URLs to prevent memory leaks
function cleanupAudioURL() {
    if (audioPlayer.src && audioPlayer.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioPlayer.src);
    }
}

// Cleanup when page is unloaded
window.addEventListener('beforeunload', cleanupAudioURL);