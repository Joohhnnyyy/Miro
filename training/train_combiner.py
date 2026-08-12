import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import spacy
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from signals.vocabulary import VocabularySignal
from signals.narrative import NarrativeSignal
from signals.stylometry import StylometrySignal
from signals.classifier import ClassifierSignal
from signals.paraphrase_resistance import ParaphraseResistanceSignal

def extract_features(df, detector):
    features = []
    labels = []
    
    # Label mapping: ai-based are 1, human are 0
    label_map = {
        'human': 0,
        'human_paraphrased': 0,
        'ai': 1,
        'ai_polished': 1,
        'ai_paraphrased': 1
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        y = label_map[row['label']]
        
        doc = detector.nlp(text)
        sentences = [s.text for s in doc.sents]
        
        # 1. Vocab density
        vocab_density, _ = detector.vocab.score_sentence(text)
        
        # 2. Narrative
        narrative_var = detector.narrative.score_essay(sentences)
        
        # 3. Stylometry
        style_features = detector.stylometry.extract_features(text)
        
        # 4. Classifier
        clf_prob = detector.classifier.score_text(text)
        
        # 5. Paraphrase Resistance
        para_features = detector.paraphrase.extract_features(doc)
        
        row_features = [
            vocab_density,
            narrative_var,
            *style_features,
            clf_prob,
            *para_features
        ]
        
        features.append(row_features)
        labels.append(y)
        
    return np.array(features), np.array(labels)

def train_combiner(data_dir, output_dir):
    print("Loading models and signals...")
    class MockDetector:
        def __init__(self):
            weights_dir = "/Users/anshjohnson/AI_Detection/backend/model_weights"
            self.nlp = spacy.load("en_core_web_sm")
            self.vocab = VocabularySignal(os.path.join(weights_dir, "signature_list.json"))
            self.narrative = NarrativeSignal()
            self.stylometry = StylometrySignal()
            self.classifier = ClassifierSignal(os.path.join(weights_dir, "final_model"))
            self.paraphrase = ParaphraseResistanceSignal()
            
    detector = MockDetector()
    
    print("Loading datasets...")
    # For speed in this demo, let's just train the combiner on the validation set or a subset of train
    # 100k takes too long to extract features on CPU for Paraphrase/Narrative.
    # We will use 2000 from train and 500 from val
    
    train_df = pd.read_json(os.path.join(data_dir, 'train.jsonl'), lines=True).sample(2000, random_state=42)
    test_df = pd.read_json(os.path.join(data_dir, 'test.jsonl'), lines=True).sample(500, random_state=42)
    
    print(f"Extracting features for {len(train_df)} training samples...")
    X_train, y_train = extract_features(train_df, detector)
    
    print(f"Extracting features for {len(test_df)} test samples...")
    X_val, y_val = extract_features(test_df, detector)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("Training Logistic Regression combiner...")
    # Balanced class weight to handle any imbalance, C parameter for regularization
    clf = LogisticRegression(class_weight='balanced', C=1.0, random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    model_path = os.path.join(output_dir, "combiner_logreg.joblib")
    scaler_path = os.path.join(output_dir, "combiner_scaler.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Combiner model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Print the explicit learned weights
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
    
    weights = clf.coef_[0]
    print("\nLearned Feature Weights (Beta):")
    for name, weight in zip(feature_names, weights):
        print(f"  {name}: {weight:+.4f}")
    print(f"  Intercept: {clf.intercept_[0]:+.4f}\n")
    
    # Evaluate
    preds = clf.predict(X_val_scaled)
    print("Validation Report:")
    print(classification_report(y_val, preds))

if __name__ == "__main__":
    train_combiner(
        data_dir="/Users/anshjohnson/AI_Detection/data/processed",
        output_dir="/Users/anshjohnson/AI_Detection/training"
    )
