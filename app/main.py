from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import os
import sys
from pathlib import Path
from sklearn.exceptions import NotFittedError

# Get the current directory (points to app/ directory)
BASE_DIR = Path(__file__).resolve().parent

# Go up one level to the project root, then into models/
MODEL_PATH = BASE_DIR.parent / "models/model.joblib"
TFIDF_PATH = BASE_DIR.parent / "models/tfidf.joblib"
MLB_PATH = BASE_DIR.parent / "models/mlb.joblib"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize models as None
model = None
tfidf = None
mlb = None

def load_models():
    global model, tfidf, mlb
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Loading tfidf from: {TFIDF_PATH}")
        print(f"Loading mlb from: {MLB_PATH}")
        
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        mlb = joblib.load(MLB_PATH)
        
        # Verify TF-IDF is fitted
        try:
            _ = tfidf.transform(["test string"])
            print("TF-IDF verified as fitted")
        except NotFittedError:
            raise ValueError("TF-IDF vectorizer is not fitted properly!")
            
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Load models when starting up
try:
    load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"FATAL: {str(e)}")
    sys.exit(1)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, question: str = Form(...)):
    if model is None or tfidf is None or mlb is None:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Models not loaded properly. Please contact administrator."
        })
    
    # Input validation
    if len(question.strip()) < 3:  # Minimum 3 characters
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error_message": "Please enter a proper question (minimum 3 characters)",
            "question": question,
            "tags": [],
            "model_name": "Naive Bayes"
        })
   
    
    try:
        # Transform and predict
        X_new = tfidf.transform([question])
        probas = model.predict_proba(X_new)
        
        # Get top 3 tags
        tags_with_conf = []
        for i, tag in enumerate(mlb.classes_):
            prob_positive = probas[i][0, 1]  # Note: This assumes predict_proba returns [n_tags, n_classes, n_samples]
            if prob_positive > 0.1:
                tags_with_conf.append((tag, round(prob_positive * 100)))
        
        tags_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate max confidence for warning
        max_conf = max([conf for (_, conf) in tags_with_conf])/100 if tags_with_conf else 0
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "question": question,
            "tags": tags_with_conf[:3],
            "model_name": "Naive Bayes",
            "max_confidence": max_conf,
            "error_message": None
        })
        
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error_message": f"Prediction failed: {str(e)}",
            "question": question,
            "tags": [],
            "model_name": "Naive Bayes"
        })

# cd /Users/beibei/ML_FINAL/app 
# uvicorn main:app --reload

#  How to merge dictionaries in Python?