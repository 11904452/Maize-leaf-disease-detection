import base64
import io
import cv2 as cv
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from DisplayDisease import DisplayDisease
import tensorflow as tf

# Create the Flask application
app = Flask(__name__)
binary_model = None
multi_model = None
segment_model = None
dd = DisplayDisease()

# Define a function to load the pre-trained models
def initialize_models():
    global binary_model, multi_model, segment_model
    binary_model = load_model('m_model.h5')
    multi_model = load_model('m_model.h5')

# Load the pre-trained models
initialize_models()
# Define a function to preprocess the binary input image
def preprocess_binary_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to preprocess the multiclass input image
def preprocess_multiClass_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/binary.html', methods=['GET', 'POST'])
def binary():
    # If the user uploads an image
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read the image file
        image = Image.open(image_file)

        # Preprocess the image
        processed_image = preprocess_binary_image(image)

        # Predict the image contents
        prediction = binary_model.predict(processed_image)
        predicted_class = np.argmax(prediction)


        # Return the prediction result as a string
        if predicted_class == 2:
            return jsonify({'result': 'Healthy'})
        else:
            return jsonify({'result': 'Diseased'})

    # If the user opens the home page
    else:
        # Return the HTML/CSS/JS code for the home page
        return render_template('binary.html')

@app.route('/multi.html', methods=['GET', 'POST'])
def multi():
    # If the user uploads an image
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read the image file
        image = Image.open(image_file)

        # Preprocess the image
        processed_image = preprocess_multiClass_image(image)

        # Predict the image contents
        prediction = multi_model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Return the prediction result as a string
        if predicted_class == 0:
            return jsonify({'result': 'Common Rust'})
        elif predicted_class == 1:
            return jsonify({'result': 'Gray Leaf Spot'})
        elif predicted_class == 2:
            return jsonify({'result': 'Healthy'})
        else:
            return jsonify({'result': 'Northern'})


    # If the user opens the home page
    else:
        # Return the HTML/CSS/JS code for the home page
        return render_template('multi.html')

@app.route('/segment.html', methods=['GET', 'POST'])
def segment():
    img_base64 = ""
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Please select an image file."
        file = request.files['image']
        if file.filename == '':
            return "Please select an image file."
        img = cv.imdecode(np.frombuffer(
            file.read(), np.uint8), cv.IMREAD_UNCHANGED)
        dd.readImage(img)
        dd.removeNoise()
        dd.displayDisease()
        img_str = cv.imencode('.jpg', dd.getImage())[1].tobytes()
        img_base64 = base64.b64encode(img_str).decode()

        return img_base64
    
    return render_template('segment.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)