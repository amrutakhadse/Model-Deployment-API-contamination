"""
FastAPI application for Iris flower classification.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define the input data model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="API for classifying Iris flowers using a Random Forest model",
    version="1.0.0"
)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")

# Define the class names
class_names = ['setosa', 'versicolor', 'virginica']

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/predict": "POST endpoint for prediction",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(iris_features: IrisFeatures):
    """
    Predict the Iris flower class based on input features.
    
    Args:
        iris_features: IrisFeatures object containing sepal_length, sepal_width, 
                      petal_length, and petal_width
    
    Returns:
        Dictionary with predicted class, class name, and probabilities
    """
    try:
        # Convert input to numpy array
        features = np.array([[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get the class name
        class_name = class_names[prediction]
        
        # Create response
        response = {
            "prediction": int(prediction),
            "class_name": class_name,
            "probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
