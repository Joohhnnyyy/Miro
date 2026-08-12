import json
import os
import re

class VocabularySignal:
    def __init__(self, signature_path):
        self.signatures = []
        if os.path.exists(signature_path):
            with open(signature_path, 'r') as f:
                self.signatures = json.load(f)
        
        # We only care about AI tells (direction == ai)
        self.ai_tells = {item['phrase']: item['z_score'] for item in self.signatures if item['direction'] == 'ai'}
        
    def score_sentence(self, sentence):
        """Scores a sentence based on density of AI signature phrases weighted by z-score."""
        text = sentence.lower()
        score = 0.0
        matches = []
        for phrase, z in self.ai_tells.items():
            # Use regex to find whole words
            if re.search(r'\b' + re.escape(phrase) + r'\b', text):
                score += z
                matches.append(phrase)
                
        # Normalize by length (a longer sentence might have more matches naturally)
        words = text.split()
        if len(words) == 0:
            return 0.0, []
        
        density = score / len(words)
        
        # Require a minimum density before it counts (as per instructions: "one flagged word alone is not a signal")
        if len(matches) < 2 and density < 5.0:
            return 0.0, matches
            
        return density, matches
