# Heart Attack Risk Prediction

A machine learning web application that predicts potential heart attack risk based on user-provided medical parameters. Built with Python, Scikit-learn, and Streamlit, this project demonstrates how a trained classification model can be integrated into an interactive application.

## Overview

This project implements a machine learning classification model to estimate heart attack risk using selected patient health indicators. Users can enter medical parameters through a web interface and receive a prediction from the trained model.

The application uses a saved preprocessing pipeline to transform input data before passing it to the trained classification model.

**Reported model accuracy:** 87%

> **Disclaimer:** This project is intended for educational and demonstration purposes only. Its predictions are not medical diagnoses and should not be used to make healthcare decisions. Consult a qualified healthcare professional for medical advice.

## Features

- **Interactive web interface** built with Streamlit.
- **Medical parameter inputs** for numerical and categorical variables.
- **Automated preprocessing** using a saved preprocessing object.
- **Machine learning inference** using a trained classification model.
- **Prediction results** displayed directly in the application.
- **Reusable model artifacts** loaded from serialized `.pkl` files.

## Tech Stack

- **Python** — application development
- **Pandas** — input data handling
- **Scikit-learn** — machine learning preprocessing and prediction
- **Joblib** — loading serialized model artifacts
- **Streamlit** — interactive web application

## Input Features

The application accepts the following 13 features:

| Feature | Description |
|---|---|
| `age` | Patient's age |
| `sex` | Sex category encoded as 0 or 1 |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar category |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina category |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels represented in the input data |
| `thal` | Thalassemia-related category |

Categorical values must follow the encoding expected by the trained preprocessing pipeline.

## Project Structure

```text
modeldeployment_AplikasiDeteksiRisikoSeranganJantung/
├── files/               # Supporting project files
├── app.py               # Streamlit application
├── model.pkl            # Trained classification model
├── preprocessing.pkl    # Saved preprocessing pipeline
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9 or a compatible version supported by the installed dependencies
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/armin-coding/modeldeployment_AplikasiDeteksiRisikoSeranganJantung.git
```

### 2. Navigate to the Project Directory

```bash
cd modeldeployment_AplikasiDeteksiRisikoSeranganJantung
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## How It Works

1. **Input collection:** The user enters the patient's medical parameters through the Streamlit interface.
2. **Data preparation:** The application converts the input values into a Pandas DataFrame with the expected feature names.
3. **Preprocessing:** The saved preprocessing pipeline transforms the input into the format expected by the model.
4. **Prediction:** The trained classification model generates a prediction.
5. **Result display:** The application displays the predicted class.

### Prediction Output

The application currently displays two possible outcomes:

- **Class 0:** Low-risk prediction.
- **Class 1:** High-risk prediction.

These labels represent the application's classification output, not a confirmed clinical assessment.

## Model Performance

The repository reports an accuracy of **0.87 (87%)**.

Accuracy represents the proportion of evaluated samples classified correctly. However, accuracy alone does not establish whether a model is suitable for medical applications.

A more complete evaluation should include:

- Precision
- Recall and sensitivity
- F1-score
- Confusion matrix
- ROC-AUC, where applicable
- Validation on independent data

In a medical risk classification setting, false negatives are particularly important to evaluate because they may represent patients whose risk is not identified by the model.

## Limitations

- Predictions depend on the quality and representativeness of the model's training data.
- The reported accuracy does not establish clinical reliability or generalizability.
- The application does not replace professional medical evaluation.
- The model's predictions should not be interpreted as individualized medical advice.
- The preprocessing pipeline and model must remain compatible with the input features and their encoding.

## Future Improvements

- Add a model evaluation report with precision, recall, F1-score, and a confusion matrix.
- Improve input validation and user-facing error handling.
- Document the training dataset and data preprocessing methodology.
- Add automated tests for the prediction pipeline.
- Provide model versioning and reproducible dependency management.
- Improve accessibility and the visual design of the interface.
- Deploy the application to a publicly accessible hosting platform, if appropriate.

## Disclaimer

This application is an educational machine learning project. It is not a certified medical device and has not been established as clinically validated. Its predictions must not be used to diagnose, treat, or rule out heart disease. Always seek professional medical evaluation for health concerns.

## Author

**Project repository:** [modeldeployment_AplikasiDeteksiRisikoSeranganJantung](https://github.com/armin-coding/modeldeployment_AplikasiDeteksiRisikoSeranganJantung)
