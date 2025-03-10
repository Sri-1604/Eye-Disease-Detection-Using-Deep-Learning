import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Ensure 'static/uploads' directory exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model("evgg.h5")

# Define disease categories
disease_classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# Define routes
@app.route('/')
def index():
    return render_template('index.html')  # Home page

@app.route('/home')
def home():
    return render_template("index.html")  # Redirect to home

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/inp')
def inp():
    return render_template("predict.html")  # Image upload page

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')  # Show upload page

    if 'image' not in request.files:
        return render_template('predict.html', prediction="No file uploaded.", image_path=None)

    file = request.files['image']
    if file.filename == '':
        return render_template('predict.html', prediction="No file selected.", image_path=None)

    # Save the uploaded file to 'static/uploads'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = np.argmax(model.predict(img_array), axis=1)[0]
    result = disease_classes[prediction]

    return render_template('predict.html', prediction=result, image_path=file.filename)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
