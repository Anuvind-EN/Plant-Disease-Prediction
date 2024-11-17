from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import webbrowser
import threading

# Flask app setup
app = Flask(__name__, template_folder='index')
app.secret_key = 'key24'
app.config["IMAGE_UPLOADS"] = "static/images/uploads"
app.config["ALLOWED_EXTENSIONS"] = {'png', 'jpg', 'jpeg'}

# Load Keras model and class names
model = tf.keras.models.load_model('Model/model.keras')
with open('Model/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process image for prediction
def process_image(img_path):
    img = Image.open(img_path).resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize and add batch dimension
    return np.expand_dims(img_array, axis=0)

# Predict disease based on image
def predict_disease(img_path):
    img_array = process_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class_index]

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/index", methods=["POST"])
def index():
    if request.files:
        image = request.files.get("image")
        if not image or not allowed_file(image.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save the image
        image_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
        image.save(image_path)
        
        # Make prediction and return result
        predicted_class = predict_disease(image_path)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"predicted_class": predicted_class})
    
    return render_template("index.html")

# Open the browser automatically
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()  # Open the browser after 1 second
    app.run(debug=True, use_reloader=False)
