# 👁️ Screen Time Eye Risk Assessment System

A smart web-based application that predicts the risk of digital eye strain using Machine Learning and Deep Learning techniques. This project combines **BiGRU (Bidirectional Gated Recurrent Unit)** for feature extraction and **Random Forest** for classification to provide accurate eye strain risk assessment based on users' screen usage behavior.

---

## 📌 Project Overview

With the increasing use of digital devices, eye strain has become a common health concern among students, professionals, and frequent screen users. This project aims to analyze screen usage patterns and predict the likelihood of eye strain while providing personalized recommendations to promote healthier screen habits.

The system collects user information such as screen time, session duration, brightness level, night usage, dark mode usage, and ambient lighting conditions. Based on these factors, it generates a risk score and displays preventive recommendations.

---

## 🎯 Objectives

* Predict eye strain risk using user screen usage patterns.
* Develop a hybrid **BiGRU + Random Forest** prediction model.
* Provide real-time risk assessment through a web interface.
* Generate personalized eye care recommendations.
* Encourage healthier digital device usage habits.

---

## 🚀 Features

### 🔐 User Authentication

* Secure login system
* Session management

### 📊 Risk Prediction

* Eye strain risk prediction using Hybrid ML Model
* Dynamic risk score generation (0–100%)

### 📈 Interactive Dashboard

* Risk gauge visualization
* Analytics dashboard
* User-friendly interface

### 🛡️ Input Validation

Strict constraints to ensure realistic inputs:

| Parameter         | Allowed Range |
| ----------------- | ------------- |
| Daily Screen Time | 0–24 Hours    |
| Night Usage       | 0–11 Hours    |
| Brightness Level  | 1–10          |
| Ambient Light     | 1–10          |

### 💡 Personalized Recommendations

* Follow 20-20-20 rule
* Reduce night-time screen exposure
* Optimize brightness settings
* Take regular breaks

---

## 🏗️ System Architecture

```text
User Input
     │
     ▼
Data Validation
     │
     ▼
Feature Engineering
     │
     ▼
Data Preprocessing
     │
     ▼
BiGRU Feature Extraction
     │
     ▼
Random Forest Classification
     │
     ▼
Risk Score Generation
     │
     ▼
Recommendations & Dashboard
```

---

## 🧠 Machine Learning Approach

### BiGRU (Bidirectional GRU)

Used for learning complex patterns and extracting meaningful representations from screen usage data.

### Random Forest

Used as the final classifier to predict eye strain risk based on extracted features.

### Feature Engineering

Additional features created include:

* Session Hour
* Interaction Feature
* Ambient Light Analysis
* Screen Usage Metrics

---

## 📂 Dataset Information

The dataset contains information related to:

* Daily Screen Time
* Average Session Duration
* Night Usage
* Brightness Level
* Dark Mode Usage
* Gender
* Ambient Light
* Eye Strain Label

The dataset was preprocessed to handle missing values, normalize features, and improve model performance.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask

### Machine Learning

* TensorFlow / Keras
* Scikit-Learn
* NumPy
* Pandas

### Visualization

* Chart.js
* Gauge Charts
* Power BI Dashboard

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 92%   |
| Precision | 92%   |
| Recall    | 92%   |
| F1 Score  | 92%   |
| AUC Score | ~0.90 |

---

## 📷 Application Screens

### Login Page

* Secure authentication interface

### Dashboard

* User input collection
* Validation checks

### Result Page

* Risk score gauge
* Risk classification
* Recommendations

### Analytics Dashboard

* Interactive visualizations
* Eye strain insights

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/eye-risk-assessment.git
cd eye-risk-assessment
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📋 Project Structure

```text
EyeStrainML/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── bigru_feature_extractor.h5
│
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   └── result.html
│
├── static/
│   ├── style.css
│   ├── gauge.js
│   ├── dashboard.pdf
│   └── images/
│
├── dataset/
│   └── eye_strain_dataset.csv
│
└── README.md
```

---

## 🔍 Challenges Faced

One of the biggest challenges was integrating the **BiGRU deep learning model with the Random Forest classifier** while maintaining feature consistency across training and deployment. Ensuring correct feature dimensions, preprocessing alignment, and model compatibility required extensive experimentation and debugging.

---

## 🌟 Future Enhancements

* Mobile application support
* Real-time eye monitoring through webcam
* Personalized health insights
* Cloud deployment
* Explainable AI integration (SHAP/LIME)
* Multi-user analytics dashboard

---

## 👨‍💻 Team

**Minor Project – Screen Time Eye Risk Assessment System**

Developed as an academic project to explore the application of Machine Learning and Deep Learning techniques in digital health monitoring and preventive eye care.

---

## 📜 License

This project is developed for educational and research purposes. Feel free to use and modify it for learning and academic work.

---

### ⭐ If you found this project useful, consider giving it a star on GitHub!
