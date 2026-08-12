import numpy as np
import os
import torch
import nltk
from nltk.corpus import wordnet as wn

class ParaphraseResistanceSignal:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_attempted = False
        # Ensure wordnet is available
        try:
            wn.synsets('dog')
        except:
            nltk.download('wordnet', quiet=True)
            
    def _get_model(self):
        if self.model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device="cpu")
                print("Signal E: Loaded SentenceTransformer onto CPU.")
            except Exception as e:
                print(f"Notice: ParaphraseResistance fallback: {e}")
                self.model = None
        return self.model
            
    def semantic_surface_gap(self, sentences):
        if len(sentences) < 3:
            return []
            
        model = self._get_model()
        if model is None:
            return []
            
        try:
            with torch.no_grad():
                embs_tensor = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
                embs = embs_tensor.cpu().numpy()
                
            gaps = []
            for i in range(1, len(sentences) - 1):
                norm_a = np.linalg.norm(embs[i])
                norm_b = np.linalg.norm(embs[i-1])
                if norm_a == 0 or norm_b == 0:
                    sem_sim = 0.0
                else:
                    sem_sim = np.dot(embs[i], embs[i-1]) / (norm_a * norm_b)
                    
                toks_a = set(sentences[i].lower().split())
                toks_b = set(sentences[i-1].lower().split())
                union_len = len(toks_a | toks_b)
                surface_overlap = len(toks_a & toks_b) / max(union_len, 1)
                
                gaps.append(float(sem_sim - surface_overlap))
            return gaps
        except Exception as e:
            print(f"Paraphrase local model error: {e}")
            
        return []
        
    def synonym_substitution_score(self, doc): # doc is a spaCy doc
        scores = []
        for token in doc:
            if token.pos_ not in {"NOUN", "VERB", "ADJ"}:
                continue
            synsets = wn.synsets(token.lemma_)
            if not synsets:
                continue
            lemma_freqs = [l.count() for syn in synsets for l in syn.lemmas()]
            if lemma_freqs and max(lemma_freqs) > 0:
                this_freq = next((l.count() for syn in synsets for l in syn.lemmas() if l.name() == token.lemma_), 0)
                scores.append(this_freq / max(lemma_freqs))
        return float(sum(scores) / len(scores)) if scores else 0.0
        
    def rhythm_autocorrelation(self, sentence_lengths):
        if len(sentence_lengths) < 3:
            return 0.0
        arr = np.array(sentence_lengths, dtype=float) - np.mean(sentence_lengths)
        if arr.std() > 0:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        return 0.0
        
    def extract_features(self, doc): # doc is a spaCy doc
        sentences = [s.text for s in doc.sents]
        
        gaps = self.semantic_surface_gap(sentences)
        avg_gap = np.mean(gaps) if gaps else 0.0
        
        syn_score = self.synonym_substitution_score(doc)
        
        sent_lengths = [len(s) for s in doc.sents]
        rhythm_ac = self.rhythm_autocorrelation(sent_lengths)
        
        return [float(avg_gap), float(syn_score), float(rhythm_ac)]
