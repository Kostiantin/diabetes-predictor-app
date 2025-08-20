# Diabetes Predictor Web App

# 🧠 Diabetes Predictor Web App

> A simple and fast machine learning web app to predict diabetes based on patient health data.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green)

---

## 🚀 Overview

This project is a lightweight, interactive **machine learning web app** that predicts the likelihood of diabetes based on user input. It leverages a **pre-trained Balanced Random Forest model**, a **FastAPI** backend, and a **HTML/Jinja2 frontend**.  

**[Try the Live Demo](https://checkdiabetes.net/)** to predict diabetes risk with your own data!


---

## 📸 Preview

![Diabetes Prediction Screenshot](assets/diabetes_prediction.png)

---

## ✨ Features

- 🧪 Model training using `BalancedRandomForestClassifier` with GridSearch
- 📊 Input form: age, BMI, glucose, hypertension, etc.
- 📦 Pre-trained model loaded via `joblib`
- ⚡ FastAPI backend with async support
- 🎨 Simple, clean HTML frontend with Jinja2 templating
- 🧠 Optional retraining using Jupyter Notebook
- 🚀 Ready for local use or cloud deployment (AWS, Render, etc.)

---

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **ML:** Scikit-learn, imbalanced-learn, joblib
- **Frontend:** HTML5, CSS, JS, Jinja2
- **Language:** Python 3.10+
- **Training Notebook:** Jupyter

---

## 📦 Project Structure

```plaintext
diabetes-predictor-app/
│
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── templates/             # HTML templates (index.html)
│   │   └── index.html
│   └── static/                # Static files (CSS, favicon, images)
│       ├── css/
│       │   └── styles.css
│       ├── favicon.png
│       
│
├── model/
│   └── diabetes_model.pkl     # Trained ML model
│
├── notebooks/
│   └── train_model.ipynb      # Training and evaluation notebook
│
├── requirements.txt
├── README.md
└── .gitignore
```
   
---

## 📋 How To Install Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Kostiantin/diabetes-predictor-app.git
   cd diabetes-predictor-app
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```

## 📋 How To Use From AWS

1. Create AWS Credentials File (if needed)
     In your project root, create `aws_credentials.py` with:

     AWS_ACCESS_KEY_ID = ""
     AWS_SECRET_ACCESS_KEY = ""
     AWS_REGION = ""
     AWS_BUCKET = ""
     AWS_MODEL_NAME = "diabetes_model.pkl"  # Update if renamed


2. Create an S3 Bucket

     Log in to AWS.
     Create a new S3 bucket.
     Set its name as the value of AWS_BUCKET in aws_credentials.py.


3. Upload the Model

     Upload diabetes_model.pkl to the S3 bucket.
     If renamed, update AWS_MODEL_NAME accordingly.

4. Create an IAM User and Role

     In the AWS IAM section, create a new user.
     Assign a role with S3 access (AmazonS3ReadOnlyAccess policy)

5. Generate Access Keys

     Generate an access key and secret key for the IAM user.
     Fill in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in aws_credentials.py.


## Train Your Own Model
Use the notebook in the `notebooks/` directory to load your data and train a new model. Save it with `joblib` or `pickle` to the `model/` folder.

---

## Author
Kostiantyn Zavizion (Your contact info or portfolio link here)