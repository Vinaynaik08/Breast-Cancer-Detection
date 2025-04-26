# Breast Cancer Detection Using Deep Learning

This project focuses on automating breast cancer detection using deep learning techniques applied to mammogram and ultrasound images. We use pre-trained convolutional neural networks (CNNs) such as **InceptionV3**, **ResNet-50**, and **VGG-16** to classify images as **Benign** or **Malignant**.

## 🧪 Project Objective

- Automate the detection of breast cancer using deep learning.
- Improve diagnostic accuracy over traditional methods.
- Compare CNN models to identify the best-performing architecture.
- Integrate the model into a web application for easy use by clinicians.

## 🛠️ Technologies Used

- **Python 3.7+**
- **TensorFlow 2.x**
- **Keras**
- **Flask** (for web backend)
- **HTML/CSS/JavaScript** (frontend)
- **Pre-trained CNNs**: InceptionV3, ResNet-50, VGG-16

## 🧬 Model Architectures & Results

| Model       | Accuracy |
|-------------|----------|
| InceptionV3 | 80%      |
| ResNet-50   | 62%      |
| VGG-16      | 57%      |

- Data was sourced from **Kaggle** and **DDSM** (Digital Database for Screening Mammography).
- Preprocessing and augmentation techniques like rotation, flipping, and zooming were used.



## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

## 📷 Web Application Preview

Upload a mammogram or ultrasound image, and the model will predict if the tumor is **Benign** or **Malignant** with a confidence score.

![Screenshot 2025-04-26 112121](https://github.com/user-attachments/assets/c777773d-9f7d-40bf-a227-b2bad04c07d3)
![Screenshot 2025-04-26 112210](https://github.com/user-attachments/assets/459893f3-4997-4388-b89a-218296637a7d)
![Screenshot 2025-04-26 112153](https://github.com/user-attachments/assets/c4febb09-7fe0-4f71-8ea5-a0b7170943a5)


## 🔍 Future Scope

- Add MRI and thermographic image support.
- Real-world clinical validation and explainability (using Grad-CAM, SHAP).
- Lightweight models for low-resource environments.
- Integration into electronic health record systems.

## 📚 References

Refer to the project report for all research citations, datasets used, and detailed explanations of methods.

DataSet

Mamogram Dataset

![5_640805896_png rf 994310cc405370dfcb9ffa8f175afc80](https://github.com/user-attachments/assets/315aecc3-7354-437a-a6f7-fc4f83f64982)
Benign
![106_76321767_png rf aa21c5be141d4ee32be960f2dfc4db8f](https://github.com/user-attachments/assets/8572040e-5d47-4bc7-bbe4-e7b13b453472)
Malignant

UltraSound DataSet

![Benign (437)](https://github.com/user-attachments/assets/672ade07-c375-4e1e-a8b0-d246f34a95ee)
Benign
![Malignant (1)](https://github.com/user-attachments/assets/cf1e74a6-0f3c-40da-b817-e912dc5decb0)
Malignant
