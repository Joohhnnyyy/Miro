import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def build_signature_list(train_path, output_dir, z_threshold=1.96, min_freq=5, conc_threshold=0.6):
    print(f"Loading {train_path}...")
    df = pd.read_json(train_path, lines=True)
    
    # We want to compare human vs AI. We can collapse the paraphrased/polished into AI/human for this signature.
    # Label mapping:
    label_map = {
        'human': 'human',
        'human_paraphrased': 'human',
        'ai': 'ai',
        'ai_polished': 'ai',
        'ai_paraphrased': 'ai'
    }
    df['binary_label'] = df['label'].map(label_map)
    
    print("Vectorizing text (unigrams + bigrams)...")
    # We DO NOT strip stopwords per master prompt instructions!
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=min_freq, stop_words=None, lowercase=True)
    X = vectorizer.fit_transform(df['text'])
    vocab = vectorizer.get_feature_names_out()
    
    # Separate by class
    ai_mask = (df['binary_label'] == 'ai').values
    human_mask = (df['binary_label'] == 'human').values
    
    X_ai = X[ai_mask]
    X_human = X[human_mask]
    
    # Frequencies
    ai_counts = np.array(X_ai.sum(axis=0)).flatten()
    human_counts = np.array(X_human.sum(axis=0)).flatten()
    
    total_ai_words = ai_counts.sum()
    total_human_words = human_counts.sum()
    
    # Prior for Dirichlet smoothing (using overall frequency)
    alpha = ai_counts + human_counts
    alpha0 = alpha.sum()
    
    print("Computing Dirichlet-smoothed log-odds and z-scores...")
    # Dirichlet smoothed log odds
    # odds_ai = (ai_counts + alpha) / (total_ai_words + alpha0 - ai_counts - alpha) # rough approx
    # Standard formula (Monroe et al.):
    # log odds delta = log( (y_ai + a_ai) / (n_ai + a_0 - y_ai - a_ai) ) - log( (y_human + a_human) / (n_human + a_0 - y_human - a_human) )
    # But usually a_ai = a0 * (alpha / alpha0), which simplifies things.
    # Let's just use a simple constant smoothing for speed and numerical stability: alpha=1
    
    y_i = ai_counts
    y_j = human_counts
    n_i = total_ai_words
    n_j = total_human_words
    
    prior = (y_i + y_j) / (n_i + n_j)
    alpha_w = prior * 1000 # scaling factor for prior
    
    odds_ai = (y_i + alpha_w) / (n_i + 1000 - y_i - alpha_w)
    odds_human = (y_j + alpha_w) / (n_j + 1000 - y_j - alpha_w)
    
    log_odds_delta = np.log(odds_ai) - np.log(odds_human)
    
    # Variance for z-score
    var_ai = 1 / (y_i + alpha_w)
    var_human = 1 / (y_j + alpha_w)
    var_delta = var_ai + var_human
    z_scores = log_odds_delta / np.sqrt(var_delta)
    
    # Candidate phrases (z >= threshold or z <= -threshold)
    # The instructions say "direction: ai", so we mainly care about AI tells (z >= threshold)
    # But we'll keep human tells too if they are strong.
    candidates_idx = np.where(np.abs(z_scores) >= z_threshold)[0]
    
    print(f"Found {len(candidates_idx)} significant n-grams.")
    
    # Topic concentration
    print("Computing topic concentration...")
    # We need to know the distribution of prompt_topic for each candidate
    # This is expensive for all candidates, so we do it only for the top N by absolute z-score
    # Sort candidates by |z_score| descending, take top 5000
    sorted_candidates_idx = candidates_idx[np.argsort(-np.abs(z_scores[candidates_idx]))]
    top_candidates = sorted_candidates_idx[:5000]
    
    # Extract topics
    topics = df['prompt_topic'].fillna('unknown').values
    unique_topics = list(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    topic_ids = np.array([topic_to_id[t] for t in topics])
    
    signature_list = []
    excluded = []
    
    # Instead of iterating rows, let's use the sparse matrix (much faster)
    for idx in tqdm(top_candidates):
        phrase = vocab[idx]
        z = z_scores[idx]
        delta = log_odds_delta[idx]
        direction = "ai" if z > 0 else "human"
        
        # Get all documents containing this phrase
        doc_indices = X[:, idx].nonzero()[0]
        if len(doc_indices) == 0:
            continue
            
        # Get topics of these documents
        doc_topics = topic_ids[doc_indices]
        
        # Count topics
        topic_counts = Counter(doc_topics)
        most_common_count = topic_counts.most_common(1)[0][1]
        concentration = most_common_count / len(doc_indices)
        
        item = {
            "phrase": phrase,
            "log_odds": float(delta),
            "z_score": float(z),
            "topic_concentration": float(concentration),
            "direction": direction
        }
        
        if concentration > conc_threshold:
            excluded.append(item)
        else:
            signature_list.append(item)
            
    print(f"Kept {len(signature_list)} phrases, excluded {len(excluded)} due to topic concentration.")
    
    # Save top 500
    # The prompt says "Keep the top ~500 surviving n-grams, ranked by z-score"
    # Wait, it means ranked by z-score magnitude for both directions? Or just AI?
    # Usually AI detection looks for AI tells. Let's sort by magnitude.
    signature_list.sort(key=lambda x: abs(x['z_score']), reverse=True)
    top_500 = signature_list[:500]
    
    os.makedirs(output_dir, exist_ok=True)
    
    sig_path = os.path.join(output_dir, "signature_list.json")
    exc_path = os.path.join(output_dir, "excluded_by_concentration.json")
    
    with open(sig_path, 'w') as f:
        json.dump(top_500, f, indent=2)
        
    with open(exc_path, 'w') as f:
        json.dump(excluded, f, indent=2)
        
    print(f"Saved {sig_path} and {exc_path}")
    print("\nTop 10 AI signatures:")
    for s in [x for x in top_500 if x['direction'] == 'ai'][:10]:
        print(f"  {s['phrase']} (z={s['z_score']:.2f}, conc={s['topic_concentration']:.2f})")

if __name__ == "__main__":
    build_signature_list(
        train_path="/Users/anshjohnson/AI_Detection/data/processed/train.jsonl",
        output_dir="/Users/anshjohnson/AI_Detection/training"
    )
