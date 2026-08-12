import kagglehub
import os
import shutil

os.environ["KAGGLE_USERNAME"] = "anshjohnson" 
os.environ["KAGGLE_KEY"] = "KGAT_357014e2ea2d41a9b295cdb254641d05"

def download_navjotkaushal():
    print("Downloading navjotkaushal dataset...")
    path = kagglehub.dataset_download("navjotkaushal/human-vs-ai-generated-essays")
    
    dest_dir = "data/raw/navjotkaushal"
    os.makedirs(dest_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(path):
        for f in files:
            src = os.path.join(root, f)
            print(f"  - {src}")
            shutil.copy(src, dest_dir)
            
    print(f"\nCopied files to {dest_dir}")

if __name__ == "__main__":
    download_navjotkaushal()
