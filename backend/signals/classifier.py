import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class ClassifierSignal:
    def __init__(self, model_dir):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Classifier signal could not load from {model_dir}: {e}")
            self.model = None
            
    def score_text(self, text):
        if self.model is None:
            return 0.5
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = F.softmax(outputs.logits, dim=-1)
        # Assuming label 1 is AI
        ai_prob = probs[0][1].item()
        return float(ai_prob)

    def score_sentences(self, sentences):
        if not sentences:
            return []
            
        if self.model is None:
            return [0.5 for _ in sentences]
            
        # Dynamic batch size based on input length, capped at 32 for memory safety
        batch_size = min(len(sentences), 32)
        results = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_texts = sentences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
                for j in range(len(batch_texts)):
                    results.append(float(probs[j][1].item()))
                    
        return results
