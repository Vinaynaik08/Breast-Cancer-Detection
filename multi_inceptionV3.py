# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set up image directories for mammogram and ultrasound
mammogram_dir = 'comb-data/mammogram/'
ultrasound_dir = 'comb-data/ultrasound/'

# Image size for InceptionV3
IMG_HEIGHT, IMG_WIDTH = 299, 299  # InceptionV3 expects 299x299 images

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
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of InceptionV3
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Output layer for 2 classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the data generator
train_generator = train_datagen.flow(x_train, y_train, batch_size=16)
model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 16,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=1
)

# Save the trained model
model.save('BreastCancerModel_InceptionV3.keras')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')