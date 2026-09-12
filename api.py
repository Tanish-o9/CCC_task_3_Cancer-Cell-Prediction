import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Cancer Prediction API",
    description="FastAPI Backend for Breast Cancer Tumor Classification",
    version="1.0"
)

# Load single model artifact dictionary (model, scaler, feature_names)
model = None
scaler = None
feature_names = []

try:
    data = joblib.load("model.pkl")
    model = data.get("model")
    scaler = data.get("scaler")
    feature_names = data.get("feature_names", [])
except Exception as e:
    print(f"Warning: model.pkl not loaded yet ({e}). Run cancer_model.ipynb first!")

# Pydantic Input Schema with realistic default tumor values
class CancerPredictionInput(BaseModel):
    mean_radius: float = Field(default=14.12, description="Cell Size (Mean Radius)")
    mean_texture: float = Field(default=19.28, description="Cell Texture (Mean Texture)")
    mean_perimeter: float = Field(default=91.96, description="Cell Perimeter (Mean Perimeter)")
    mean_area: float = Field(default=654.88, description="Cell Area (Mean Area)")
    mean_smoothness: float = Field(default=0.096, description="Cell Smoothness")
    mean_compactness: float = Field(default=0.104, description="Cell Density / Compactness")
    mean_concavity: float = Field(default=0.088, description="Cell Shape Depth")
    mean_concave_points: float = Field(default=0.048, description="Cell Corner Points")
    mean_symmetry: float = Field(default=0.181, description="Cell Shape Symmetry")
    mean_fractal_dimension: float = Field(default=0.062, description="Cell Edge Complexity")

@app.get("/")
def home():
    return {"message": "Cancer Cell Classifier API is running!", "docs_url": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "Healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_cancer(data: CancerPredictionInput):
    if model is None or scaler is None or not feature_names:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded!")

    # Collect inputs into dictionary
    inputs = data.model_dump()
    
    # Build full feature array matching feature_names
    full_features = {}
    for col in feature_names:
        key = col.replace(" ", "_")
        if key in inputs:
            full_features[col] = inputs[key]
        elif col == "radius_to_perimeter":
            full_features[col] = inputs["mean_radius"] / (inputs["mean_perimeter"] + 1e-5)
        elif col == "compactness_to_smoothness":
            full_features[col] = inputs["mean_compactness"] / (inputs["mean_smoothness"] + 1e-5)
        else:
            full_features[col] = inputs.get("mean_" + col.replace("worst ", "").replace("error ", ""), 0.0)

    # Convert to DataFrame to retain column names
    input_df = pd.DataFrame([full_features])[feature_names]

    # Scale features & Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probabilities))
        malignant_prob = float(probabilities[0])
        benign_prob = float(probabilities[1])
    else:
        confidence = 1.0
        malignant_prob = 1.0 if prediction == 0 else 0.0
        benign_prob = 1.0 if prediction == 1 else 0.0

    result = "Safe (No Cancer / Benign)" if prediction == 1 else "Cancerous Tumor (Harmful / Malignant)"

    return {
        "prediction": result,
        "class_id": int(prediction),
        "confidence_score": round(confidence * 100, 2),
        "probabilities": {
            "Benign": round(benign_prob * 100, 2),
            "Malignant": round(malignant_prob * 100, 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
