import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).parent
TEST_FILE = BASE_DIR / "test.csv"
CM_FILE = BASE_DIR / "confusion_matrix.png"

def evaluate(run_id):
    # Load test data
    test_data = pd.read_csv(TEST_FILE)
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Load model dari MLflow pakai run_id
    model_uri = f"runs:/{run_id}/sklearn-model"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_test)

    # Hitung Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Bikin file PNG buat Confusion Matrix (sesuai ipynb)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Low Risk (0)', 'High Risk (1)'], 
                yticklabels=['Low Risk (0)', 'High Risk (1)'])
    plt.title("Heart Attack Risk Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save gambar sementara pakai path dinamis
    plt.savefig(CM_FILE)
    plt.close()

    # Resume MLflow run pakai run_id buat nge-log metrics & gambar CM
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_artifact(str(CM_FILE)) # Log gambarnya jadi artifact

    print(f"✅ Evaluation complete.")
    print(f"Accuracy: {accuracy:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

    # Return Recall (kalau nanti mau dipakai buat logic lain di pipeline)
    return recall 

if __name__ == "__main__":
    # Ini buat ngetes doang, string "your_run_id_here" diganti pas dijalanin beneran
    # evaluate("your_run_id_here")
    pass