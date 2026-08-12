import os
import requests
import time

class ClassifierSignal:
    def __init__(self, model_dir=None):
        self.api_url = "https://api-inference.huggingface.co/models/neo2454132/miro-detector-model"
        self.api_token = os.environ.get("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        
        if not self.api_token:
            print("Warning: HF_TOKEN not found in environment. Inference API may be rate limited or fail.")

    def _query(self, payload, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=20)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading
                    estimated_time = response.json().get("estimated_time", 10)
                    print(f"Model is loading, waiting {estimated_time} seconds...")
                    time.sleep(min(estimated_time, 20))
                else:
                    print(f"Error from HF API (Status {response.status_code}): {response.text}")
                    time.sleep(2)
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(2)
        return None

    def score_text(self, text):
        if not text:
            return 0.5
            
        result = self._query({"inputs": text})
        
        if result and isinstance(result, list) and len(result) > 0:
            # Handle both [[{...}, {...}]] and [{...}, {...}]
            items = result[0] if isinstance(result[0], list) else result
            
            ai_prob = 0.0
            found_labels = False
            for label_score in items:
                if isinstance(label_score, dict):
                    label = str(label_score.get("label", "")).lower()
                    if label in ["ai", "ai_polished", "ai_paraphrased", "label_1"]:
                        ai_prob += float(label_score.get("score", 0))
                        found_labels = True
                        
            if found_labels:
                return min(1.0, ai_prob)
                    
        return 0.5

    def score_sentences(self, sentences):
        if not sentences:
            return []
            
        results = []
        # Free API limits batch sizes
        batch_size = 10
        
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i:i+batch_size]
            api_result = self._query({"inputs": batch_texts})
            
            if api_result and isinstance(api_result, list):
                for s_result in api_result:
                    ai_prob = 0.0
                    found_labels = False
                    
                    # Handle both [[{...}], [{...}]] and [{...}]
                    items = s_result if isinstance(s_result, list) else [s_result]
                    for label_score in items:
                        if isinstance(label_score, dict):
                            label = str(label_score.get("label", "")).lower()
                            if label in ["ai", "ai_polished", "ai_paraphrased", "label_1"]:
                                ai_prob += float(label_score.get("score", 0))
                                found_labels = True
                                
                    results.append(min(1.0, ai_prob) if found_labels else 0.5)
            else:
                results.extend([0.5] * len(batch_texts))
                
        return results
