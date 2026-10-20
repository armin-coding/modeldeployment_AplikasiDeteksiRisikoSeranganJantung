import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = Path(__file__).parent
TRAIN_FILE = BASE_DIR / "train.csv"

def train():
    # Setup nama eksperimen di MLflow
    mlflow.set_experiment("Heart-Attack-Prediction")

    # Load preprocessed training data
    train_data = pd.read_csv(TRAIN_FILE)
    
    # Pisahin features dan label
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Mulai tracking MLflow
    with mlflow.start_run() as run:
        
        # Setup parameter sesuai ipynb lu
        params = {
            "random_state": 42, 
            "max_iter": 1000
        }
        
        # Log parameter ke MLflow biar gampang di-track di dashboard
        mlflow.log_params(params)

        # Train modelnya
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Log & Register the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="HeartAttackLogReg"
        )
        


        joblib.dump(model, BASE_DIR / "model.pkl")
        print(f"✅ Training complete. Model logged with Run ID: {run.info.run_id}")
        
        # Return Run ID buat dipakai di evaluation.py nanti
        return run.info.run_id 

if __name__ == "__main__":
    train()