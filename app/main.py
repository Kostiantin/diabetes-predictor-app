from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import os
import tempfile
import boto3

    
#to remove previous configs and cached model
#os.remove(os.path.join(tempfile.gettempdir(), "diabetes_model.pkl"))

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

USE_LOCAL_MODEL = 0 # 0 - use from AWS

# ------------------------
# Model Loading Logic
# ------------------------

def load_model_from_s3():
    
    print("loading from aws")
    
    try:
        
        # in root directory you need to have this file aws_credentials.py with all constants
        
        from aws_credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_BUCKET, AWS_MODEL_NAME
        
    except ImportError:
        
        AWS_ACCESS_KEY_ID = None
        AWS_SECRET_ACCESS_KEY = None
        AWS_REGION = None
        AWS_BUCKET = None
        AWS_MODEL_NAME = None
    
    """Download model from S3 and load it."""
    s3 = boto3.client("s3",
      aws_access_key_id=AWS_ACCESS_KEY_ID,
      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
      region_name=AWS_REGION
    )
    
    tmp_dir = tempfile.gettempdir()
    local_model_path = os.path.join(tmp_dir, AWS_MODEL_NAME)

    # Download from S3 if not already cached
    if not os.path.exists(local_model_path):
        s3.download_file(AWS_BUCKET, AWS_MODEL_NAME, local_model_path)

    return joblib.load(local_model_path)

def load_model():
    """Decide whether to use local model or S3 model."""
    use_local = USE_LOCAL_MODEL # 1 - from AWS, 0 - local model
    if use_local:
        
        print("loading from local")
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(base_dir, "model", "diabetes_model.pkl")
        return joblib.load(model_path)
    else:
        return load_model_from_s3()

# Load model at startup
model = load_model()

# ------------------------
# Preprocessing
# ------------------------
def preprocess_input(gender, age, bmi, hba1c, glucose, heart_disease, hypertension, smoking_history):
    # One-hot encode gender
    gender_Female = 1 if gender == "Female" else 0
    gender_Male = 1 if gender == "Male" else 0
    gender_Other = 1 if gender == "Other" else 0

    # One-hot encode smoking_history
    smoking_options = ["No Info", "current", "ever", "former", "never", "not current"]
    smoking_encoded = {f"smoking_history_{opt}": int(smoking_history == opt) for opt in smoking_options}

    # Binary encode
    heart_disease_flag = 1 if heart_disease == "Yes" else 0
    hypertension_flag = 1 if hypertension == "Yes" else 0

    # Build dictionary
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

    # Ensure correct order
    column_order = [
        'age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level',
        'gender_Female','gender_Male','gender_Other',
        'smoking_history_No Info','smoking_history_current','smoking_history_ever',
        'smoking_history_former','smoking_history_never','smoking_history_not current'
    ]

    return pd.DataFrame([[data[col] for col in column_order]], columns=column_order)

# ------------------------
# Routes
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "diabetic_prob": None,
        "form_data": {}
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

    df = preprocess_input(gender, age, bmi, hba1c, glucose, heart_disease, hypertension, smoking_history)
    pred_proba = model.predict_proba(df)
    pred_class = "Diabetic" if pred_proba[0][1] >= 0.5 else "Not Diabetic"
    result = f"Prediction: {pred_class} (Probability of Diabetic: {pred_proba[0][1]:.2%})"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "diabetic_prob": round(pred_proba[0][1] * 100, 2),
        "form_data": form_data
    })
