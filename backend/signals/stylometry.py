import spacy
import textstat
import numpy as np

class StylometrySignal:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            # Fallback if model not downloaded
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
            
        self.transition_phrases = {
            "moreover", "furthermore", "however", "therefore", "thus", "consequently", 
            "nevertheless", "on the other hand", "in conclusion", "to summarize", 
            "importantly", "additionally", "firstly", "secondly", "in addition"
        }
        
    def extract_features(self, text):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return [0.0]*6
            
        # 1. Sentence length variance
        sent_lengths = [len(s) for s in sentences]
        len_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0.0
        
        # 2. Type-token ratio
        words = [token.text.lower() for token in doc if token.is_alpha]
        if len(words) == 0:
            ttr = 0.0
        else:
            ttr = len(set(words)) / len(words)
            
        # 3. Transition phrase density
        text_lower = text.lower()
        transition_count = sum(text_lower.count(p) for p in self.transition_phrases)
        trans_density = transition_count / max(len(sentences), 1)
        
        # 4. Passive voice ratio
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass" or token.dep_ == "csubjpass")
        passive_ratio = passive_count / max(len(sentences), 1)
        
        # 5. Punctuation diversity
        punct = [token.text for token in doc if token.is_punct]
        if punct:
            punct_div = len(set(punct)) / len(punct)
        else:
            punct_div = 0.0
            
        # 6. Readability score
        readability = textstat.flesch_reading_ease(text)
        
        return [
            float(len_variance), 
            float(ttr), 
            float(trans_density), 
            float(passive_ratio), 
            float(punct_div), 
            float(readability)
        ]
