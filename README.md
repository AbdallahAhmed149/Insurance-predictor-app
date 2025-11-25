# Insurance Cost Predictor App

A web application that predicts insurance charges based on personal and demographic information, using a machine learning model trained on historical insurance data.

---

## 🚀 Features

- **Interactive Web UI**: User-friendly interface with sliders and dropdowns for all inputs.
- **Machine Learning Powered**: Utilizes a scikit-learn linear regression model for predictions.
- **Live Prediction**: Instantly estimates insurance costs when entering details.
- **Feature Engineering**: Advanced handling of input data, including age and BMI groupings, one-hot encoding, and interaction terms.
- **Robust Backend**: Flask-powered API for model inference.

---

## 🏗️ Project Structure

```
Insurance-predictor-app/
├── data/
│   └── insurance.csv                # Training data
├── models/
│   ├── insurance_model.pkl          # Saved trained model
│   └── onehot_encoder.pkl           # Saved one-hot encoder
├── templates/
│   └── gui.html                     # Frontend (HTML + Tailwind CSS + JS)
├── app.py                           # Backend (Flask app)
├── model.py                         # Model training & feature engineering
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🌐 Usage

### 1. Install Dependencies

You'll need Python 3.7+ and [pip](https://pip.pypa.io/en/stable/).
Install dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

If you want to retrain the model, run:

```bash
python model.py
```

This will generate the `.pkl` files (`insurance_model.pkl`, `onehot_encoder.pkl`) in the `models/` directory.

### 3. Run the Application

Start the Flask server:

```bash
python app.py
```

By default, the app will be available at [http://localhost:5000](http://localhost:5000).

---

## 🌍 Live Application

Try the app instantly: [https://abdallahahmed.pythonanywhere.com/](https://abdallahahmed.pythonanywhere.com/)

---

## 🖥️ Web Demo

- Open your browser and go to `http://localhost:5000`.
- Fill out the details (age, gender, BMI, children, smoker status, and region).
- Click **Predict Insurance Cost** to see your estimated annual charge.

---

## ⚙️ Model Details

- **Algorithm**: Linear Regression (scikit-learn)
- **Dataset**: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Preprocessing**:
  - Log transformation of charges
  - One-hot encoding of categorical variables (`sex`, `region`)
  - Feature engineering: interaction terms, age/BMI groupings
  - Outlier clipping

---

## 📁 Key Files

- `app.py` – Flask server that loads the trained model and serves predictions.
- `model.py` – Script for model training and feature engineering.
- `templates/gui.html` – Frontend interface, uses Tailwind CSS for modern styling.
- `models/insurance_model.pkl` – Pretrained model, loaded by the backend.
- `models/onehot_encoder.pkl` – Categorical feature encoder for inference.
- `data/insurance.csv` – Original training data (from Kaggle).

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

---

## ✨ Credits

- Built by [Abdallah Ahmed](https://github.com/AbdallahAhmed149)
- Dataset by [MIRICHOI0218](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Powered by Python, Flask, scikit-learn, Tailwind CSS

---

## 📬 Issues & Contributions

Feel free to [open issues](https://github.com/AbdallahAhmed149/Insurance-predictor-app/issues) or submit pull requests to contribute!
