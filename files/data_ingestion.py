import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "Heart Attack Data Set.csv"
INGESTED_DIR = BASE_DIR / "ingested"
OUTPUT_FILE = INGESTED_DIR / "heart_attack.csv"

def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(RAW_FILE)
    assert not df.empty, "Dataset is empty"
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data ingested from {RAW_FILE.name} → {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()