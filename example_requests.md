# Example API Requests and Responses

## Example 1: Setosa Iris

### Request
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

### Response
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

## Example 2: Versicolor Iris

### Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.2,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
  }'
```

### Response
```json
{
  "prediction": 1,
  "class_name": "versicolor",
  "probabilities": {
    "setosa": 0.0,
    "versicolor": 0.97,
    "virginica": 0.03
  },
  "confidence": 0.97
}
```

## Example 3: Virginica Iris

### Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 7.7,
    "sepal_width": 3.0,
    "petal_length": 6.1,
    "petal_width": 2.3
  }'
```

### Response
```json
{
  "prediction": 2,
  "class_name": "virginica",
  "probabilities": {
    "setosa": 0.0,
    "versicolor": 0.02,
    "virginica": 0.98
  },
  "confidence": 0.98
}
```

## Health Check

### Request
```bash
curl -X GET "http://localhost:8000/health"
```

### Response
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Python Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Sample data for different Iris types
samples = [
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },
    {
        "sepal_length": 6.2,
        "sepal_width": 2.9,
        "petal_length": 4.3,
        "petal_width": 1.3
    },
    {
        "sepal_length": 7.7,
        "sepal_width": 3.0,
        "petal_length": 6.1,
        "petal_width": 2.3
    }
]

# Make predictions
for i, sample in enumerate(samples, 1):
    response = requests.post(url, json=sample)
    result = response.json()
    print(f"Sample {i}: Predicted class - {result['class_name']} (confidence: {result['confidence']:.2f})")
```

### Output
```
Sample 1: Predicted class - setosa (confidence: 1.00)
Sample 2: Predicted class - versicolor (confidence: 0.97)
Sample 3: Predicted class - virginica (confidence: 0.98)
```
