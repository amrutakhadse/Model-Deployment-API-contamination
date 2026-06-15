# Task 3: Model Deployment - API & Containerization

This project demonstrates model deployment using FastAPI and Docker containerization. It includes a trained Random Forest classifier for Iris flower classification wrapped in a REST API.

## Project Structure

```
.
├── app.py              # FastAPI application with inference endpoint
├── train_model.py      # Script to train and save the model
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration for containerization
├── .dockerignore       # Files to exclude from Docker build
├── model.joblib        # Trained model (generated after running train_model.py)
└── README.md           # This file
```

## Setup Instructions

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train_model.py
   ```
   This will create a `model.joblib` file with the trained Random Forest classifier.

3. **Run the API:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will be available at `http://localhost:8000`

### Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t iris-classifier-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 iris-classifier-api
   ```
   Open http://localhost:8000/docs in your browser to test the prediction endpoint interactively
   The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Root endpoint with API information.

### GET `/health`
Health check endpoint to verify the API is running and the model is loaded.

### POST `/predict`
Predict the Iris flower class based on input features.

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "confidence": 1.0
}
```

## Example Requests

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(url, json=data)
print(response.json())
```

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Model Details

- **Algorithm:** Random Forest Classifier
- **Dataset:** Iris dataset (150 samples, 4 features, 3 classes)
- **Classes:** setosa, versicolor, virginica
- **Features:** sepal_length, sepal_width, petal_length, petal_width
- **Training Accuracy:** ~1.0
- **Test Accuracy:** ~1.0

## Technologies Used

- **FastAPI:** Modern, fast web framework for building APIs
- **Uvicorn:** ASGI server for running FastAPI
- **Scikit-learn:** Machine learning library for model training
- **Docker:** Containerization platform for deployment
- **Pydantic:** Data validation using Python type annotations
