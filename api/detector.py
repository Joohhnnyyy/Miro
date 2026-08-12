import os
import xgboost as xgb
import numpy as np
import spacy

from signals.vocabulary import VocabularySignal
from signals.narrative import NarrativeSignal
from signals.stylometry import StylometrySignal
from signals.classifier import ClassifierSignal
from signals.paraphrase_resistance import ParaphraseResistanceSignal

class Detector:
    def __init__(self, base_dir=None):
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print("Loading signals...")
        if base_dir is None:
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
        else:
            weights_dir = os.path.join(base_dir, "backend", "model_weights")
        self.vocab = VocabularySignal(os.path.join(weights_dir, "signature_list.json"))
        self.narrative = NarrativeSignal()
        self.stylometry = StylometrySignal()
        self.classifier = ClassifierSignal(os.path.join(weights_dir, "final_model"))
        self.paraphrase = ParaphraseResistanceSignal()
        
        import joblib
        print("Loading combiner model...")
        
        self.combiner = None
        self.scaler = None
        self.is_logreg = False
        
        lr_path = os.path.join(weights_dir, "combiner_logreg.joblib")
        scaler_path = os.path.join(weights_dir, "combiner_scaler.joblib")
        
        if os.path.exists(lr_path) and os.path.exists(scaler_path):
            self.combiner = joblib.load(lr_path)
            self.scaler = joblib.load(scaler_path)
            self.is_logreg = True
        else:
            combiner_path = os.path.join(weights_dir, "combiner_xgboost.model")
            self.combiner = xgb.Booster()
            if os.path.exists(combiner_path):
                self.combiner.load_model(combiner_path)
            else:
                print("WARNING: Combiner model not found.")
            
    def detect(self, text):
        doc = self.nlp(text)
        sentences = [s.text for s in doc.sents]
        
        vocab_density, matches = self.vocab.score_sentence(text)
        narrative_var = self.narrative.score_essay(sentences)
        style_features = self.stylometry.extract_features(text)
        clf_prob = self.classifier.score_text(text)
        para_features = self.paraphrase.extract_features(doc)
        
        features = [
            vocab_density,
            narrative_var,
            *style_features,
            clf_prob,
            *para_features
        ]
        
        feature_names = [
            "Vocab Density",
            "Narrative Variance",
            "Length Variance",
            "Type-Token Ratio",
            "Transition Density",
            "Passive Ratio",
            "Punctuation Div",
            "Readability",
            "Classifier Prob (DeBERTa)",
            "Semantic Surface Gap",
            "Synonym Sub Score",
            "Rhythm Autocorr"
        ]
        
        feature_contributions = []
        
        if self.is_logreg and self.scaler:
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            ai_prob = float(self.combiner.predict_proba(X_scaled)[0][1])
            
            weights = self.combiner.coef_[0]
            scaled_vals = X_scaled[0]
            for name, val, weight, raw in zip(feature_names, scaled_vals, weights, features):
                feature_contributions.append({
                    "name": name,
                    "value": float(raw),
                    "scaled_value": float(val),
                    "weight": float(weight),
                    "contribution": float(val * weight)
                })
        else:
            try:
                dmatrix = xgb.DMatrix(np.array([features]))
                ai_prob = float(self.combiner.predict(dmatrix)[0])
            except:
                ai_prob = clf_prob
                
        # Clamp probability to prevent overconfidence (Veritas strategy)
        ai_prob = min(max(round(float(ai_prob), 4), 0.01), 0.99)
        
        # Categorical band for heatmap
        if ai_prob > 0.70:
            band = "high_ai"
            band_label = "AI-Skewed"
        elif ai_prob > 0.35:
            band = "uncertain"
            band_label = "Mixed / Uncertain"
        else:
            band = "human"
            band_label = "Human-Like"
                
        analyzed_sentences = []
        
        # Batch evaluate valid sentences
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        valid_probs = self.classifier.score_sentences(valid_sentences)
        prob_idx = 0
        
        for s in sentences:
            if len(s.strip()) > 10:
                s_prob = valid_probs[prob_idx]
                prob_idx += 1
                
                s_prob = min(max(round(float(s_prob), 4), 0.01), 0.99)
                
                if s_prob >= 0.70:
                    s_band = "high_ai"
                elif s_prob >= 0.40:
                    s_band = "uncertain"
                else:
                    s_band = "human"
                    
                analyzed_sentences.append({
                    "text": s,
                    "ai_probability": s_prob,
                    "is_ai": s_prob > 0.5,
                    "band": s_band
                })
            else:
                analyzed_sentences.append({
                    "text": s,
                    "ai_probability": 0.5,
                    "is_ai": False,
                    "band": "uncertain"
                })

        total_sents = max(len(analyzed_sentences), 1)
        ai_sents = sum(1 for s in analyzed_sentences if s["band"] == "high_ai")
        human_sents = sum(1 for s in analyzed_sentences if s["band"] == "human")
        unc_sents = total_sents - ai_sents - human_sents
        
        human_ratio = human_sents / total_sents
        ai_ratio = ai_sents / total_sents
        
        # Authenticity is roughly human ratio + half of uncertain ratio
        authenticity_score = human_ratio + ((unc_sents / total_sents) * 0.5)

        return {
            "ai_probability": ai_prob,
            "authenticity_score": authenticity_score,
            "sentence_distribution": {
                "human_ratio": human_ratio,
                "ai_ratio": ai_ratio,
                "human_count": human_sents,
                "ai_count": ai_sents,
                "uncertain_count": unc_sents
            },
            "is_ai": ai_prob >= 0.70,
            "band": band,
            "band_label": band_label,
            "feature_contributions": feature_contributions,
            "sentences": analyzed_sentences,
            "signals": {
                "vocabulary_density": vocab_density,
                "ai_phrases_found": matches,
                "narrative_variance": narrative_var,
                "stylometry": {
                    "sentence_length_variance": style_features[0],
                    "type_token_ratio": style_features[1],
                    "transition_density": style_features[2],
                    "passive_ratio": style_features[3],
                    "punctuation_diversity": style_features[4],
                    "readability": style_features[5]
                },
                "classifier_probability": clf_prob,
                "paraphrase_resistance": {
                    "semantic_surface_gap": para_features[0],
                    "synonym_substitution_score": para_features[1],
                    "rhythm_autocorrelation": para_features[2]
                }
            }
        }
