# AI-Based Waste Segregation System

An AI-powered real-time waste classification system built with **MobileNetV2**, **TensorFlow/Keras**, and **Computer Vision** techniques to automatically classify waste materials into distinct recyclable categories.

The system is capable of classifying waste into:
- Metal
- Paper / Cardboard
- Plastic
- Trash


This project also supports **real-time webcam-based prediction** using **OpenCV + DroidCam integration**, making it a step toward intelligent recycling and automated waste management systems.

---

#  Features

-  Real-time waste classification using webcam input
- Transfer Learning with MobileNetV2  
- Deep Learning-based image classification  
- Confusion Matrix evaluation  
- Accuracy & Loss visualisation graphs  
- Organised training and inference pipeline  
- Lightweight model suitable for real-time applications  
- Future-ready for robotics and smart recycling systems

---

# Model Information

| Component | Details |
|---|---|
| Model Architecture | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Technique | Transfer Learning |
| Classes | Metal, Paper, Plastic, Trash |
| Validation Accuracy | ~88% |
| Prediction Mode | Real-Time Webcam Classification |

---

# Training Performance

## Training vs Validation Accuracy

<img width="1385" height="1012" alt="image" src="https://github.com/user-attachments/assets/1c4b7902-2316-495d-815f-5992fa64a267" />


---

## Training vs Validation Loss

<img width="1210" height="1002" alt="image" src="https://github.com/user-attachments/assets/8fde5b72-5184-40c4-ab07-4cf550e1fa6e" />


---

# Confusion Matrix

<img width="1230" height="1013" alt="image" src="https://github.com/user-attachments/assets/09752440-6300-4dcf-ab50-819b9950d69f" />


The confusion matrix shows strong classification performance across all four waste categories with balanced prediction capability.

---

# Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Project Structure

waste-segregation-ai/
│
├── model/                    # Trained model files
├── notebooks/                # Training notebooks and evaluation graphs
├── src/                      # Prediction scripts and utilities
├── test_camera.py            # Real-time webcam prediction script
├── requirements.txt          # Required Python dependencies
├── README.md
└── .gitignore

---

# System Workflow
1) Dataset collection and preprocessing
2) Image augmentation and normalisation
3) Transfer learning using MobileNetV2
4) Model training and validation
5) Performance evaluation using a confusion matrix
6) Real-time prediction using webcam integration

---

# Real-Time Camera Prediction

This project supports real-time waste classification using:
1) DroidCam
2) OpenCV webcam feed
3) Live prediction overlay
4) Run the camera prediction system using:
        python test_camera.py

---

# Installation & Setup

1) Clone Repository
git clone https://github.com/Maddy152006/waste-segregation-ai.git

2) Move Into Project Directory
cd waste-segregation-ai

3) Install Dependencies
pip install -r requirements.txt

4) Run Real-Time Prediction
python test_camera.py

---

# Dataset

The dataset is not included in this repository due to size limitations.
Download Dataset Here:
   https://drive.google.com/drive/folders/1lz4Efzse1HAXAPS96VuKsnnGvCUOucrO?usp=sharing
Dataset Classes
1) Metal
2) Paper
3) Plastic
4) Trash

---

# Future Improvements

- Robotic waste sorting arm integration
- Edge-device deployment using Raspberry Pi
- Multi-object waste detection
- Smart recycling analytics dashboard
- Faster inference optimization
- Smart city waste management integration

---

# Applications

- Smart Recycling Systems
- Automated Waste Management
- AI-Based Environmental Monitoring
- Robotics & Industrial Automation
- Smart City Infrastructure

---

# Author
Madhavan A Ramanujan
B.Tech CSE (AI & Robotics) — VIT Chennai
- GitHub:
     https://github.com/Maddy152006
- LinkedIn:
     https://linkedin.com/in/madhavan-a-ramanujan-74629b374

---

 # Project Highlights
- Real-time AI-powered waste classification
- Transfer learning using MobileNetV2
- Organised ML project structure
- Strong validation accuracy (~88%)
- Webcam-based live prediction support
- Designed for future robotics integration


