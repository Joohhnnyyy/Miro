import kagglehub
import os
import shutil

os.environ["KAGGLE_USERNAME"] = "anshjohnson" 
os.environ["KAGGLE_KEY"] = "KGAT_357014e2ea2d41a9b295cdb254641d05"

def download_aide():
    print("Downloading aide dataset...")
    path = kagglehub.dataset_download("lburleigh/tla-lab-ai-detection-for-essays-aide-dataset")
    
    dest_dir = "data/raw/aide"
    os.makedirs(dest_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(path):
        for f in files:
            src = os.path.join(root, f)
            print(f"  - {src}")
            shutil.copy(src, dest_dir)
            
    print(f"\nCopied files to {dest_dir}")

if __name__ == "__main__":
    download_aide()
