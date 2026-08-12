import numpy as np
import torch
import os

class NarrativeSignal:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_attempted = False
        
    def _get_model(self):
        if self.model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device="cpu")
                print("Signal B: Loaded SentenceTransformer onto CPU.")
            except Exception as e:
                print(f"Notice: NarrativeSignal fallback: {e}")
                self.model = None
        return self.model
        
    def score_essay(self, sentences):
        """
        Computes the variance of cosine similarities between consecutive sentences.
        Low variance -> AI-like
        High variance -> human-like
        """
        if len(sentences) < 2:
            return 0.0
            
        model = self._get_model()
        if model is None:
            return 0.05
            
        try:
            with torch.no_grad():
                embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
                norms = torch.norm(embeddings, dim=1, keepdim=True)
                norm_embeddings = embeddings / (norms + 1e-8)
                
                sims_tensor = (norm_embeddings[:-1] * norm_embeddings[1:]).sum(dim=1)
                similarities = sims_tensor.cpu().numpy().tolist()
                
            if len(similarities) == 0:
                return 0.0
            return float(np.var(similarities))
        except Exception as e:
            print(f"Narrative local model error: {e}")
            
        return 0.05
