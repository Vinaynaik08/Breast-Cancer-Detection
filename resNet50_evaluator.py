# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set up image directories for mammogram and ultrasound
mammogram_dir = 'comb-data/mammogram/'
ultrasound_dir = 'comb-data/ultrasound/'

# Image size for ResNet50
IMG_HEIGHT, IMG_WIDTH = 224, 224  # ResNet50 expects 224x224 images

# Helper function to load images from a directory
def load_images_from_folder(folder):
    dataset = []
    labels = []
    for label_value, label_name in enumerate(['no', 'yes']):
        label_folder = os.path.join(folder, label_name)
        for image_name in os.listdir(label_folder):
            if image_name.endswith('.png') or image_name.endswith('.jpg'):
                image_path = os.path.join(label_folder, image_name)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                dataset.append(image_array)
                labels.append(label_value)
    return np.array(dataset), np.array(labels)

# Load mammogram and ultrasound images
mammogram_data, mammogram_labels = load_images_from_folder(mammogram_dir)
ultrasound_data, ultrasound_labels = load_images_from_folder(ultrasound_dir)

# Combine datasets
dataset = np.concatenate((mammogram_data, ultrasound_data), axis=0)
labels = np.concatenate((mammogram_labels, ultrasound_labels), axis=0)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

# Normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Load the saved ResNet50 model
model = tf.keras.models.load_model('BreastCancerModel_ResNet50.keras')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Print evaluation results
print(f"Evaluation Loss: {loss:.4f}")
print(f"Evaluation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
