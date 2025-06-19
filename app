from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('cifar_cnn.h5')

label_map = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
        <h2>Upload an image (CIFAR-10)</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <input type="submit" value="Predict">
        </form>
    ''')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        x = preprocess_image(img_bytes)
        preds = model.predict(x)
        label = label_map[int(np.argmax(preds))]
        confidence = float(np.max(preds))
        return render_template_string(f'''
            <h2>Prediction Result</h2>
            <p><strong>Label:</strong> {label}</p>
            <p><strong>Confidence:</strong> {confidence:.4f}</p>
            <a href="/">Try another image</a>
        ''')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

