import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime
import base64
from io import BytesIO



# Flask Setup
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and preprocessors
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_order = joblib.load("feature_order.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)
    features = {
        "length": len(y),
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms_mean": np.mean(librosa.feature.rms(y=y)),
        "rms_var": np.var(librosa.feature.rms(y=y)),
        "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y)),
        "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y)),
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0]
    }

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
        features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])

    features_df = pd.DataFrame([features])
    for feature in feature_order:
        if feature not in features_df.columns:
            features_df[feature] = 0
    features_df = features_df[feature_order]
    
    return features_df

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            try:
                features_df = extract_features(filepath)
                features_scaled = scaler.transform(features_df)
                prediction = model.predict(features_scaled)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = np.max(prediction) * 100

                
                return redirect(url_for("result", genre=predicted_label, confidence=confidence))
            except Exception as e:
                return render_template("upload.html", error=str(e))
    return render_template("upload.html")


@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Extract features and make prediction
        features_df = extract_features(file_path)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        predicted_genre = label_encoder.inverse_transform([np.argmax(prediction)])
        # confidence = np.max(prediction) * 100
        
        
        return render_template('result.html', 
                               genre=predicted_genre[0], 
                            #    confidence=confidence, 
                               filename=filename)  # ✅ Pass filename here
    


@app.route("/about")
def about():
    return render_template("about.html")

import plotly
import plotly.express as px
import plotly.graph_objs as go
import json

@app.route('/dashboard/<filename>')
def dashboard(filename):
    file_path = os.path.join('static/uploads', filename)
    
    y, sr = librosa.load(file_path, duration=30)
    
    # Waveform
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    waveform_path = os.path.join('static', 'waveform.png')
    plt.savefig(waveform_path)
    plt.close()

    # Spectrogram
    plt.figure(figsize=(12, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join('static', 'spectrogram.png')
    plt.savefig(spectrogram_path)
    plt.close()

    # MFCCs
    plt.figure(figsize=(12, 3))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    mfcc_path = os.path.join('static', 'mfcc.png')
    plt.savefig(mfcc_path)
    plt.close()

    # Zero-Crossing Rate
    plt.figure(figsize=(12, 3))
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    plt.plot(zero_crossings[0])
    plt.title('Zero-Crossing Rate')
    zcr_path = os.path.join('static', 'zcr.png')
    plt.savefig(zcr_path)
    plt.close()

    # Chroma Features
    plt.figure(figsize=(12, 3))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chroma Features')
    chroma_path = os.path.join('static', 'chroma.png')
    plt.savefig(chroma_path)
    plt.close()

    # Spectral Contrast
    plt.figure(figsize=(12, 3))
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    librosa.display.specshow(spectral_contrast, x_axis='time', cmap='coolwarm')
    plt.colorbar(label='Spectral Contrast')
    plt.title('Spectral Contrast')
    spectral_contrast_path = os.path.join('static', 'spectral_contrast.png')
    plt.savefig(spectral_contrast_path)
    plt.close()

    # Spectral Flatness
    plt.figure(figsize=(12, 3))
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    plt.plot(spectral_flatness, color='green')
    plt.title('Spectral Flatness Over Time')
    plt.xlabel('Frames')
    plt.ylabel('Flatness')
    spectral_flatness_path = os.path.join('static', 'spectral_flatness.png')
    plt.savefig(spectral_flatness_path)
    plt.close()

    


    # Tonnetz (Tonal Centroid)
    plt.figure(figsize=(12, 3))
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    librosa.display.specshow(tonnetz, x_axis='time')
    plt.colorbar()
    plt.title('Tonnetz (Tonal Centroid)')
    tonnetz_path = os.path.join('static', 'tonnetz.png')
    plt.savefig(tonnetz_path)
    plt.close()

    # RMS Energy
    plt.figure(figsize=(12, 3))
    rms = librosa.feature.rms(y=y)
    plt.plot(rms[0])
    plt.title('RMS Energy')
    rms_path = os.path.join('static', 'rms.png')
    plt.savefig(rms_path)
    plt.close()

    # Tempo Curve
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    plt.figure(figsize=(12, 3))
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)  
    plt.plot(times, onset_env, label="Onset Strength", color='b')
    plt.scatter(librosa.frames_to_time(beats, sr=sr), onset_env[beats], color='r', label="Beats", marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Strength')
    plt.title(f'Tempo Curve (Estimated Tempo: {int(tempo)} BPM)')
    plt.legend()
    tempo_path = os.path.join('static', 'tempo.png')
    plt.savefig(tempo_path)
    plt.close()

    # Harmonic vs. Percussive Components
    plt.figure(figsize=(12, 3))
    harmonic, percussive = librosa.effects.hpss(y)
    plt.plot(harmonic, label="Harmonic", alpha=0.7, color='blue')
    plt.plot(percussive, label="Percussive", alpha=0.7, color='red')
    plt.title('Harmonic vs Percussive Components')
    plt.legend()
    harmonic_percussive_path = os.path.join('static', 'harmonic_percussive.png')
    plt.savefig(harmonic_percussive_path)
    plt.close()
    

    # Spectral Roll-off Curve
    plt.figure(figsize=(12, 3))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    plt.plot(rolloff[0], color='orange')
    plt.title('Spectral Roll-off')
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    spectral_rolloff_path = os.path.join('static', 'spectral_rolloff.png')
    plt.savefig(spectral_rolloff_path)
    plt.close()

    



    return render_template('dashboard.html', filename=filename,
                           waveform=waveform_path,
                           spectrogram=spectrogram_path,
                           mfcc=mfcc_path,
                           zcr=zcr_path,
                           chroma=chroma_path,
                           spectral_contrast=spectral_contrast_path,
                           spectral_flatness=spectral_flatness_path,
                           tonnetz=tonnetz_path,
                           rms=rms_path,
                           tempo=tempo_path,
                           harmonic_percussive=harmonic_percussive_path,
                           spectral_rolloff=spectral_rolloff_path)

if __name__ == "__main__":
    app.run(debug=True)
