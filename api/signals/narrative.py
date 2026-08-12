import numpy as np
from sentence_transformers import SentenceTransformer

class NarrativeSignal:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Small, local, CPU-fine model per instructions
        self.embedder = SentenceTransformer(model_name)
        
    def score_essay(self, sentences):
        """
        Computes the variance of cosine similarities between consecutive sentences.
        Low variance -> AI-like
        High variance -> human-like
        """
        if len(sentences) < 2:
            return 0.0
            
        embs = self.embedder.encode(sentences)
        
        similarities = []
        for i in range(1, len(sentences)):
            norm_a = np.linalg.norm(embs[i])
            norm_b = np.linalg.norm(embs[i-1])
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = np.dot(embs[i], embs[i-1]) / (norm_a * norm_b)
            similarities.append(sim)
            
        variance = np.var(similarities)
        return float(variance)
