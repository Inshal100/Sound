from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import json
import librosa
import soundfile
import numpy as np
import pickle
from datetime import date

# Load parameters from config.json
with open('config.json', 'r') as c:
    params = json.load(c)['params']

# Initialize Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = params['database_uri']
db = SQLAlchemy(app)

# Function to extract audio features
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        # Ensure the signal length is at least 2048
        if len(X) < 2048:
            raise ValueError("Input signal is too short for feature extraction")

        # Compute the Short-Time Fourier Transform (STFT)
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=min(len(X), 2048)))
        
        result = np.array([])
        
        # Extract MFCC features
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        # Extract Chroma features
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        
        # Extract Mel Spectrogram features
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    
    return result

# Database model for storing contact details
class Details(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(30), nullable=False)
    ph_num = db.Column(db.String(20), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(10), nullable=True)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html', params=params)

# Route for uploading and processing audio files
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return "No audio file uploaded", 400
        
        audio = request.files['audio']
        audio_path = os.path.join('uploads', audio.filename)
        audio.save(audio_path)

        # Load the pre-trained model
        with open('sound_recog.txt', 'rb') as f:
            model = pickle.load(f)

        try:
            # Extract features and predict emotion
            features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
            emotion = model.predict([features])[0]
        except ValueError as e:
            return str(e), 400

        return redirect(url_for('result', emotion=emotion))
    return render_template('upload.html', params=params)

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html', params=params)

# Route for the Contact page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        
        # Create a new entry in the database
        entry = Details(name=name, email=email, ph_num=phone, message=message, date=date.today())
        db.session.add(entry)
        db.session.commit()
    
    return render_template('contact.html', params=params)

# Route for displaying the result
@app.route('/result')
def result():
    emotion = request.args.get('emotion')
    
    # Map emotion to corresponding image
    emotion_image = {
        'neutral': '/static/images/neutral.jpg',
        'calm': '/static/images/calm.jpg',
        'happy': '/static/images/happy.jpg',
        'sad': '/static/images/sad.jpg',
        'angry': '/static/images/angry.jpg',
        'fearful': '/static/images/fearful.jpg',
        'disgust': '/static/images/disgust.jpg',
        'surprised': '/static/images/surprised.jpg',
    }.get(emotion, '/static/images/default.png')

    return render_template('output.html', emotion=emotion, emotion_image=emotion_image, params=params)

# Main entry point for the application
if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
