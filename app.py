
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import base64  

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'  
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'} 
MODEL_PATH = 'deepfake_detector_resnet_lstm.keras' # Path to your Keras model

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size (adjust as needed)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    MODEL_LOADED = True
    print("Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")


# --- Constants (taken from your training script - adjust if different) ---
IMG_SIZE = (224, 224)
FRAME_COUNT = 30

# --- Functions from your script (adjust slightly for Flask context if needed) ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame.astype(np.float32)
    frame = tf.keras.applications.resnet50.preprocess_input(frame)
    return frame

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // FRAME_COUNT)

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)
    cap.release()

    # Pad with blank frames if needed
    blank_frame = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    blank_frame = tf.keras.applications.resnet50.preprocess_input(blank_frame)
    while len(frames) < FRAME_COUNT:
        frames.append(blank_frame)

    return np.array(frames[:FRAME_COUNT])


def classify_video(video_path, model):
    try:
        frames = extract_frames(video_path)
        frames = np.expand_dims(frames, axis=0) # Add batch dimension

        prediction = model.predict(frames, verbose=0)[0][0]
        predicted_label = "Fake" if prediction > 0.5 else "Real"
        return predicted_label, prediction
    except Exception as e:
        print(f"Error during video classification: {e}")
        return "Error", -1 # Indicate an error

# --- Custom Jinja Filter ---
def format_probability(probability):
    """Formats a probability value as a percentage string."""
    return f"{probability * 100:.2f}%"

app.jinja_env.filters['format_probability'] = format_probability

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if not MODEL_LOADED:
        return render_template('index.html', model_error=True)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'video' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['video']
        # If the user does not select a file, browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            label, probability = classify_video(video_path, model)

            # Optional: Display the video itself (base64 encoding for browser) - adjust if needed
            video_display_data = None
            # with open(video_path, 'rb') as video_file:
            #     video_content = video_file.read()
            #     video_display_data = base64.b64encode(video_content).decode('utf-8')

            os.remove(video_path) # Clean up uploaded file after processing

            if label == "Error":
                 return render_template('results.html', error_processing=True) # Display error on results page
            else:
                return render_template('results.html', label=label, probability=probability, video_data=video_display_data)

        else:
            return render_template('index.html', error='Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS))

    return render_template('index.html', model_error=False) # Initial page load


if __name__ == '__main__':
    app.run(debug=True) # Use debug=False in production# Use debug=False in production