# American Sign Language (ASL) Alphabet Recognition Using MobileNetV2

## 📌 Project Overview

This project focuses on recognizing American Sign Language (ASL) alphabets (A–Z) using Deep Learning and Computer Vision techniques.

The system uses a **MobileNetV2 Convolutional Neural Network (CNN)** model to classify hand gestures into their corresponding alphabet letters. The project aims to support communication by providing a foundation for automated sign language recognition.

The model is trained using a self-collected dataset of ASL hand gesture images and evaluated using various performance metrics.

## 🎯 Objectives

- To develop an ASL alphabet recognition system using Deep Learning.
- To recognize 26 static hand gestures representing letters A–Z.
- To apply Transfer Learning using the MobileNetV2 architecture.
- To evaluate model performance using test accuracy, precision, recall, and F1-score.
- To explore real-time hand gesture recognition using a webcam.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** MobileNetV2
- **Image Processing:** OpenCV
- **Data Processing:** NumPy
- **Data Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook / VS Code

## 📂 Project Structure

```text
ASL-Recognition-Using-MobileNetV2/
│
├── dataset/
│
├── processed_dataset/
│
├── American sign language.ipynb
│
├── train_mobilenetv2.py
├── test_mobilenetv2.py
├── check_model.py
│
├── realtime.py
├── realtime_mobilenetv2.py
│
├── asl_model.keras
├── asl_mobilenetv2.keras
│
├── class_names.txt
│
├── .gitignore
└── README.md
```

## 📊 Dataset Information

- **Dataset Source:** Self-collected
- **Total Images:** 5,253
- **Number of Classes:** 26 (A–Z)
- **Image Size:** 128 × 128 pixels

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training | 3,669 |
| Validation | 782 |
| Testing | 802 |
| **Total** | **5,253** |

The dataset is divided into training, validation, and testing sets to train the model, monitor its performance, and evaluate its ability to classify unseen images.

## 🧠 Model Architecture

The project uses **MobileNetV2**, a lightweight and efficient CNN architecture designed for image classification.

Transfer Learning is used to take advantage of pretrained MobileNetV2 features.

The model training process includes:

1. Loading and preprocessing ASL alphabet images.
2. Resizing images to 128 × 128 pixels.
3. Applying the MobileNetV2 architecture.
4. Adding classification layers for 26 alphabet classes.
5. Training the model using the Adam optimizer.
6. Fine-tuning the model to improve classification performance.
7. Evaluating the trained model using test images.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Input Image Size | 128 × 128 |
| Number of Classes | 26 |
| Batch Size | 32 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Initial Training Epochs | 10 |
| Fine-tuning Epochs | 15 |
| Dropout | 0.4 |
| Validation Split | 20% |

## 📈 Model Evaluation Results

The trained model was evaluated using the test dataset.
Evaluation Metric	Result
Number of Classes	26
Test Images	802
Test Accuracy	99.63%
Test Loss	0.0124
Correct Predictions	799
Incorrect Predictions	3
Macro Precision	99.65%
Macro Recall	99.63%
Macro F1-Score	99.63%
A confusion matrix was also implemented to analyze classification performance across the 26 alphabet classes.
🎥 Real-Time Recognition

The project also includes scripts for webcam-based recognition.

The real-time recognition process involves:

Accessing the webcam using OpenCV.

Capturing video frames.

Identifying the hand gesture region.

Preprocessing the captured image.

Passing the image to the trained MobileNetV2 model.

Displaying the predicted alphabet.
 Limitations

The system focuses on static ASL alphabet gestures rather than continuous sign language sentences.

Recognition performance may vary under different lighting conditions and backgrounds.

Hand positioning, camera quality, and gesture variations can affect predictions.

Webcam recognition may not achieve the same accuracy as offline test-set evaluation.

🚀 Future Enhancements

Improve real-time recognition accuracy.

Implement hand detection and gesture segmentation.

Expand recognition to dynamic signs and complete words.

Develop a user-friendly graphical interface.

Explore integration with speech-to-text and text-to-speech systems.
