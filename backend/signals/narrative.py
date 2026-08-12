import numpy as np
import os
import requests

class NarrativeSignal:
    def __init__(self):
        pass
        
    def score_essay(self, sentences):
        """
        Computes the variance of cosine similarities between consecutive sentences.
        Low variance -> AI-like
        High variance -> human-like
        """
        if len(sentences) < 2:
            return 0.0
            
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: HF_TOKEN missing. Narrative variance fallback.")
            return 0.05
            
        url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = requests.post(url, headers=headers, json={"inputs": sentences}, timeout=10)
            if response.status_code == 200:
                embs = np.array(response.json())
                
                similarities = []
                for i in range(1, len(sentences)):
                    norm_a = np.linalg.norm(embs[i])
                    norm_b = np.linalg.norm(embs[i-1])
                    if norm_a == 0 or norm_b == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(embs[i], embs[i-1]) / (norm_a * norm_b)
                    similarities.append(sim)
                    
                if len(similarities) == 0:
                    return 0.0
                return float(np.var(similarities))
        except Exception as e:
            print(f"Narrative HF API Error: {e}")
            
        return 0.05
