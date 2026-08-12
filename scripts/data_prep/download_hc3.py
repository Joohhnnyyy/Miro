import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download

def download_hc3():
    print("Downloading HC3 dataset directly from HuggingFace Hub...")
    # The dataset has an 'all.jsonl' file in the root
    try:
        file_path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="all.jsonl", repo_type="dataset")
        print(f"Downloaded to {file_path}")
    except Exception as e:
        print(f"Failed to download all.jsonl: {e}")
        return

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            topic = item.get("question", "unknown")
            
            # Add human answers
            human_answers = item.get("human_answers", [])
            for ans in human_answers:
                if ans.strip():
                    rows.append({"text": ans, "label": "human", "prompt_topic": topic, "source": "hc3"})
                    
            # Add AI answers
            chatgpt_answers = item.get("chatgpt_answers", [])
            for ans in chatgpt_answers:
                if ans.strip():
                    rows.append({"text": ans, "label": "ai", "prompt_topic": topic, "source": "hc3"})

    df = pd.DataFrame(rows)
    
    dest_dir = "data/raw/hc3"
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, "hc3.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    download_hc3()
