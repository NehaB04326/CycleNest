from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("pcos_ultrasound_model.h5")
IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img_array = preprocess_image(img_file)
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence < 0.5:
        result = "⚠️ PCOS Detected"
        conf_display = 1.0 - confidence
        tips = "Eat a balanced diet, exercise regularly, reduce stress, and consult a gynecologist."
    else:
        result = "✅ Normal"
        conf_display = confidence
        tips = "Keep up a healthy lifestyle and monitor your wellness."

    return jsonify({'result': result, 'confidence': conf_display, 'tips': tips})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

