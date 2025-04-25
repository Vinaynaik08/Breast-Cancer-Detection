import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load both models
model_png = load_model('BreastCancerModel2.keras')
model_jpg = load_model('BreastCancerModel2_mammo.keras')

# Define the image dimensions and test directory
IMG_SIZE = 64
test_dir = 'test-30'

# Prepare lists for true labels and predictions
y_true = []
y_pred = []

# Label mapping
label_mapping = {'no': 0, 'yes': 1}

# Process each folder ('no' for benign, 'yes' for malignant)
for label_folder in ['no', 'yes']:
    label = label_mapping[label_folder]
    folder_path = os.path.join(test_dir, label_folder)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Only proceed if the file is a JPG or PNG image
        if file_extension not in ['.jpg', '.png']:
            continue

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Select the appropriate model
        if file_extension == '.png':
            prediction = model_png.predict(img)
        elif file_extension == '.jpg':
            prediction = model_jpg.predict(img)
        
        # Get the predicted label (0 for benign, 1 for malignant)
        pred_label = np.argmax(prediction)
        
        # Append to lists
        y_true.append(label)
        y_pred.append(pred_label)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Print results
print("Evaluation Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")