import os
import torch
from typing import List, Dict, Any

class ClassifierSignal:
    def __init__(self, model_path: str = "backend/model_weights/final_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
                self.model.eval()
                print(f"Signal D DeBERTa model loaded from {self.model_path} onto {self.device}")
            except Exception as e:
                print(f"Error loading trained DeBERTa model: {e}")
                self.model = None
        else:
            print(f"Notice: Trained DeBERTa model not yet present at {self.model_path}. Ready for training.")

    def score_sentences(self, sentences: List[str]) -> List[float]:
        """Predicts AI probability per sentence."""
        if not sentences:
            return []
            
        if self.model is None or self.tokenizer is None:
            # Fallback baseline heuristic if model is not yet loaded
            return [0.50 for _ in sentences]
            
        results = []
        # Batch inference
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_texts = sentences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                for j in range(len(batch_texts)):
                    # Depending on label mappings. The config.json has:
                    # 'ai', 'ai_polished', 'ai_paraphrased' at indices 1, 2, 3
                    # 'human', 'human_paraphrased' at indices 0, 4
                    p = probs[j]
                    if p.shape[0] == 5:
                        ai_prob = float(p[1].item() + p[2].item() + p[3].item())
                    else:
                        ai_prob = float(p[1].item()) if p.shape[0] > 1 else 0.5
                    results.append(ai_prob)
                    
        return results

    def score_text(self, text: str) -> float:
        res = self.score_sentences([text])
        return res[0] if res else 0.5
