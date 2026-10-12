# Insurance Cost Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML%20%2B%20Tailwind%20CSS-06B6D4.svg?logo=html5&logoColor=white)](https://tailwindcss.com/)

A full-stack machine learning application that estimates annual insurance charges from personal and demographic inputs. The system combines a Flask backend, a browser-based interface, and a trained regression model to deliver fast, usable predictions from historical insurance data.

> Live demo: https://abdallahahmed.pythonanywhere.com/

---

## Overview

Insurance cost varies based on multiple interacting factors, including age, body mass index (BMI), smoking status, family size, sex, and region. This project provides an end-to-end solution for exploring those variables and generating a practical insurance cost estimate.

The application is intentionally lightweight and transparent:

1. Users provide demographic and lifestyle inputs through a clean web form.
2. The browser sends the values to the Flask API.
3. The backend applies the same preprocessing steps used during training.
4. A serialized regression model produces the prediction.
5. The response is converted back to the original scale and shown in the UI.

This is a functional ML prototype for estimation and educational use, designed to demonstrate predictive modeling in a real product-style workflow.

---

## Key Features & Metrics

### Core Features

- **Interactive web interface** for entering age, BMI, children, smoker status, sex, and region.
- **Machine learning inference** powered by a trained scikit-learn `LinearRegression` model.
- **Feature engineering** for age bins, BMI bins, smoker interaction effects, and encoded categorical inputs.
- **REST-style prediction API** exposed through a Flask endpoint.
- **Persistent model deployment** using serialized `joblib` artifacts.
- **Real-time user feedback** with validation and response handling inside the frontend.

### Performance and Model Notes

- **Model type:** Linear Regression
- **Training approach:** 80/20 train/test split with 5-fold cross-validation
- **Prediction format:** returns a rounded annual cost estimate in USD
- **Repository composition:** approximately **55.6% HTML**, **42.0% Python**, and **2.4% Shell**
- **Deployment status:** production-style live demo available at the project URL above

> The repository currently includes the full training pipeline and model artifacts, but quantitative claims such as latency, throughput, or exact accuracy should be validated against the current training environment before publication in marketing or product documentation.

---

## System Architecture

The project follows a simple client-server inference architecture:

```text
┌──────────────────────────────┐
│ Browser UI                  │
│ templates/gui.html          │
│ HTML + Tailwind + JavaScript│
└──────────────┬──────────────┘
               │ POST /predict
               ▼
┌──────────────────────────────┐
│ Flask App                   │
│ app.py                      │
│ Request validation          │
│ Feature preprocessing       │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Trained ML Model            │
│ models/insurance_model.pkl  │
│ models/onehot_encoder.pkl   │
│ LinearRegression + encoding │
└──────────────────────────────┘
```

### Design Principles

- **Training and inference parity:** the transformation logic is replicated in both `model.py` and `app.py`.
- **Simple architecture:** a lightweight Flask application with a single prediction endpoint.
- **Artifact-based deployment:** the model and encoder are persisted and loaded at runtime.
- **Separation of concerns:** model training logic is isolated from web request handling.

---

## Tech Stack

| Category | Tools |
|---|---|
| Frontend | HTML, JavaScript, Tailwind CSS |
| Backend | Python, Flask, Flask-CORS |
| Machine Learning | scikit-learn, NumPy, pandas |
| Data Science | Matplotlib, Seaborn |
| Model Persistence | joblib |
| Deployment | Gunicorn-compatible Flask service |

---

## Getting Started

### Prerequisites

- Python 3.7+
- `pip`
- Git

### 1. Clone the repository

```bash
git clone https://github.com/AbdallahAhmed149/Insurance-predictor-app.git
cd Insurance-predictor-app
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The app will be available at:

```text
http://localhost:5000
```

### 5. Model artifacts

The repository expects the following files to exist in `models/`:

```text
models/insurance_model.pkl
models/onehot_encoder.pkl
```

If the model is retrained, `model.py` regenerates these artifacts automatically.

---

## Usage

### Web Interface

1. Open `http://localhost:5000` in your browser.
2. Fill in the required fields.
3. Click **Predict Insurance Cost**.
4. View the estimated annual premium.

### API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sex": "male",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast"
  }'
```

Example response:

```json
{
  "success": true,
  "prediction": 1234.56
}
```

### Retrain the Model

```bash
python model.py
```

This script loads the dataset, performs preprocessing and feature engineering, trains the regression model, and saves the model artifacts back into the `models/` directory.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Validate the app locally.
5. Submit a pull request with a clear summary of the change.

Please keep changes focused and ensure the project remains easy to run and understand.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

Built by [Abdallah Ahmed](https://github.com/AbdallahAhmed149).
