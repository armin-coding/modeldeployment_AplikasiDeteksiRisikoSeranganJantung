from data_ingestion import ingest_data
from preprocessing import preprocess
from train import train
from evaluation import evaluate

# We now check Recall (Sensitivity) instead of Accuracy
# Karena ini prediksi medis, Recall jauh lebih krusial buat meminimalisir False Negatives
RECALL_THRESHOLD = 0.85 

def run_pipeline():
    print("--- Step 1: Data Ingestion ---") 
    ingest_data()

    print("\n--- Step 2: Preprocessing ---")
    preprocess()

    print("\n--- Step 3: Training ---")
    run_id = train()   

    print("\n--- Step 4: Evaluation ---")
    recall = evaluate(run_id)

    print("\n--- Pipeline Decision ---")
    if recall >= RECALL_THRESHOLD:
        print(f"🟢 Model APPROVED for deployment (Recall: {recall:.3f} >= {RECALL_THRESHOLD})")
    else:
        print(f"🔴 Model REJECTED (Recall: {recall:.3f} < {RECALL_THRESHOLD})")

if __name__ == "__main__":
    run_pipeline()