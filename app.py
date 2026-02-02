# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os

app = FastAPI(title="Model Inference API")

MODEL_PATH = os.getenv("MODEL_PATH", "./model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = e

class PredictRequest(BaseModel):
    # Accept either a list of features for a single sample
    # or a batch as list of lists via `instances`.
    features: Optional[List[float]] = None
    instances: Optional[List[List[float]]] = None

class PredictResponse(BaseModel):
    predictions: List
    details: Optional[dict] = None

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    # Build input array
    if req.features is not None:
        X = np.array([req.features])
    elif req.instances is not None:
        X = np.array(req.instances)
    else:
        raise HTTPException(status_code=400, detail="Provide `features` or `instances` in the JSON body")

    # Prediction
    try:
        preds = model.predict(X).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # If model supports predict_proba, include it
    details = {}
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X).tolist()
            details["probabilities"] = probs
        except Exception:
            pass

    return {"predictions": preds, "details": details}
# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "model.pkl")
print("Saved example model to model.pkl")





