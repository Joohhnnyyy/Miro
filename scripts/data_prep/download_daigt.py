import kagglehub
import os
import shutil

# Authenticate via environment variables
os.environ["KAGGLE_USERNAME"] = "anshjohnson" # Guessing from path, will fail if wrong
os.environ["KAGGLE_KEY"] = "KGAT_357014e2ea2d41a9b295cdb254641d05"

try:
    print("Downloading DAIGT V2 dataset...")
    path = kagglehub.dataset_download("thedrcat/daigt-v2-train-dataset")
    print(f"Downloaded to: {path}")
    
    dest_dir = "data/raw/daigt_v2"
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Files in downloaded dataset:")
    for root, dirs, files in os.walk(path):
        for f in files:
            src = os.path.join(root, f)
            print(f"  - {src}")
            shutil.copy(src, dest_dir)
            
    print(f"\nCopied files to {dest_dir}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
