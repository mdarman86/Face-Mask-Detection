# 😷 Face Mask Detection with Live Alert System

## 📌 Objective

Detect whether a person is wearing a face mask in real-time using a webcam.

---

## 🧠 Overview

This project uses Computer Vision and Deep Learning to classify faces as **Mask** or **No Mask**.
It captures live video using OpenCV, detects faces using Haar Cascade, and predicts mask usage using a trained CNN model.

---

## 🛠️ Tools & Technologies

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Haar Cascade Classifier

---

## 📂 Project Structure

```
Face-Mask-Detection/
│
├── .gitignore
├── README.md
├── train.py
├── detect.py
├── Face_Mask_Detection_Report.pdf
├── mask-detection-demo.mp4
```

---

## ⚙️ Installation

```bash
pip install opencv-python tensorflow numpy scikit-learn
```

---

## ▶️ How to Run

### 1. Train Model

```bash
python train.py
```

### 2. Run Detection

```bash
python detect.py
```

Press **q** to exit webcam.

---

## 🎯 Features

* Real-time face detection
* Mask / No Mask classification
* Live webcam monitoring
* Alert system for No Mask

---

## 📸 Output

* 🟢 Green Box → Mask
* 🔴 Red Box → No Mask

---

## 📹 Demo Video

`https://drive.google.com/file/d/167nrqjM4T9tbJBJPRnIb_E7FWXQin4xF/view?usp=drive_link`

---

## 🚀 Future Improvements

* Add sound alert system 🔊
* Improve accuracy using advanced models
* Deploy using Flask

---

## 👨‍💻 Author

**Md Arman**

---

⭐ If you like this project, give it a star!
