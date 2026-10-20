import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

BASE_DIR = Path(__file__).parent
INGESTED_FILE = BASE_DIR / "ingested" / "heart_attack.csv"

def preprocess():
    # Load data hasil ingestion
    df = pd.read_csv(INGESTED_FILE)
    
    # Setup kolom numerik dan kategorik (SESUAI IPYNB)
    numcols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    catcols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    target = "target"

    # Ubah tipe data kategorik jadi string
    df[catcols] = df[catcols].astype(str)
    
    # Pisahin features dan target
    X = df[numcols + catcols]
    y = df[target]

    # Split data (stratify = best practice)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Setup Preprocessor pakai ColumnTransformer
    numeric_transformer = StandardScaler()
    # Pake sparse_output=False biar gampang digabungin jadi DataFrame lagi nanti
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numcols),
            ('cat', categorical_transformer, catcols)
        ])

    # Fit dan Transform (TRAIN di fit_transform, TEST di transform aja)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Ambil nama kolom baru hasil One Hot Encoding
    feature_names = preprocessor.get_feature_names_out()

    # Balikin ke wujud DataFrame biar bisa di-save ke CSV
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # Gabungin lagi sama label targetnya (y_train dan y_test)
    train = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    # Save ke CSV
    train.to_csv(BASE_DIR / "train.csv", index=False)
    test.to_csv(BASE_DIR / "test.csv", index=False)
    

    joblib.dump(preprocessor, BASE_DIR / "preprocessing.pkl")
    print("✅ Preprocessing complete. Scaled & Encoded train and test files saved.")

if __name__ == "__main__":
    preprocess()