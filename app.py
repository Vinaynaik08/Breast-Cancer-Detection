from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load both models
model_png = load_model('BreastCancerModel2.keras')
model_jpg = load_model('BreastCancerModel2_mammo.keras')

# Define the image dimensions
IMG_SIZE = 64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Get the file extension and save the file
        file_extension = os.path.splitext(file.filename)[1].lower()
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Determine which model to use based on file extension
        if file_extension == '.png':
            prediction = model_png.predict(img)
        elif file_extension == '.jpg':
            prediction = model_jpg.predict(img)
        else:
            return render_template('index.html', prediction="Unsupported file type. Please upload a PNG or JPG image.")

        # Interpret the prediction result
        result = 'Malignant' if np.argmax(prediction) == 1 else 'Benign'
        return render_template('index.html', prediction=result)
    
    return render_template('index.html', prediction='')

if __name__ == '__main__':
    # Create 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
