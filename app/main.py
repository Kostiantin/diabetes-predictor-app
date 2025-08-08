from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "diabetes_model.pkl")
model = joblib.load(model_path)

def preprocess_input(gender, age, bmi, hba1c, glucose, heart_disease, hypertension, smoking_history):
    # One-hot encode gender
    gender_Female = 1 if gender == "Female" else 0
    gender_Male = 1 if gender == "Male" else 0
    gender_Other = 1 if gender == "Other" else 0

    # One-hot encode smoking_history
    smoking_options = [
        "No Info", "current", "ever", "former", "never", "not current"
    ]
    smoking_encoded = {f"smoking_history_{opt}": int(smoking_history == opt) for opt in smoking_options}

    # Binary encode heart disease and hypertension
    heart_disease_flag = 1 if heart_disease == "Yes" else 0
    hypertension_flag = 1 if hypertension == "Yes" else 0

    # Build the full dictionary
    data = {
        "age": int(age),
        "hypertension": hypertension_flag,
        "heart_disease": heart_disease_flag,
        "bmi": int(bmi),
        "HbA1c_level": int(hba1c),
        "blood_glucose_level": int(glucose),
        "gender_Female": gender_Female,
        "gender_Male": gender_Male,
        "gender_Other": gender_Other,
        **smoking_encoded
    }

    # Ensure the correct order of columns
    column_order = [
        'age',
        'hypertension',
        'heart_disease',
        'bmi',
        'HbA1c_level',
        'blood_glucose_level',
        'gender_Female',
        'gender_Male',
        'gender_Other',
        'smoking_history_No Info',
        'smoking_history_current',
        'smoking_history_ever',
        'smoking_history_former',
        'smoking_history_never',
        'smoking_history_not current'
    ]

    return pd.DataFrame([[data[col] for col in column_order]], columns=column_order)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "diabetic_prob": None,
        "form_data": {}  # Initialize empty form_data for initial load
    })

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    age: int = Form(...),
    bmi: int = Form(...),
    hba1c: int = Form(...),
    glucose: int = Form(...),
    heart_disease: str = Form(...),
    hypertension: str = Form(...),
    smoking_history: str = Form(...)
):
    # Collect form data to pass back to template
    form_data = {
        "gender": gender,
        "age": age,
        "bmi": bmi,
        "hba1c": hba1c,
        "glucose": glucose,
        "heart_disease": heart_disease,
        "hypertension": hypertension,
        "smoking_history": smoking_history
    }

    # Preprocess input for prediction
    df = preprocess_input(gender, age, bmi, hba1c, glucose, heart_disease, hypertension, smoking_history)
    pred_proba = model.predict_proba(df)
    pred_class = "Diabetic" if pred_proba[0][1] >= 0.5 else "Not Diabetic"
    result = f"Prediction: {pred_class} (Probability of Diabetic: {pred_proba[0][1]:.2%})"

    # Render template with prediction and form data
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "diabetic_prob": round(pred_proba[0][1] * 100, 2),
        "form_data": form_data
    })