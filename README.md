# AI Admissions Essay Detector
**Protecting Academic Integrity through Adversarial Resilience and Linguistic Forensics**

## Overview
Traditional AI detectors rely on basic probability mapping, making them highly vulnerable to simple paraphrase attacks—like running a ChatGPT essay through Quillbot. This project introduces a next-generation AI forensics engine built not just to catch raw AI generation, but to expose the adversarial laundering, paraphrasing, and humanizing attacks that bypass traditional detectors. 

Our system achieves a highly robust **99% binary accuracy** (Human vs. AI) across out-of-domain data, while actively identifying and documenting vulnerabilities in edge-case adversarial classes.

---

## The 5-Signal Architecture
Our platform processes every submission through a multi-layered forensic pipeline, analyzing the text across five independent dimensions before a final XGBoost Combiner makes the ultimate judgment.

1. **Deep Learning Classification (Signal D)**: *The Neural Engine.* We utilized a fine-tuned `distilbert-base-uncased` model. It scans for the deep, high-dimensional signatures unique to Large Language Models.
2. **Paraphrase Resistance Analysis**: *The Launder-Catcher.* Measures "Semantic Surface Gap" and "Rhythm Autocorrelation." Paraphrasing destroys natural sentence flow; this signal catches the mechanical choppiness left behind.
3. **Narrative Variance Scoring**: *The Predictability Check.* Human writers naturally vary sentence lengths and structural cadence. This signal measures the mathematical predictability of the essay's narrative flow.
4. **Forensic Stylometry**: *The Grammar Fingerprint.* Extracts hard linguistic features (Type-Token Ratio, passive voice frequency, punctuation diversity) to compare against known human baselines.
5. **AI Vocabulary Density**: *The Tell-Tale Lexicon.* Scans for statistically anomalous concentrations of known AI transitional phrases (e.g., "Furthermore," "In conclusion").

---

## Dataset Provenance

| Source / Dataset | Total Rows | Contribution | Notes & Dropped Data |
| :--- | :--- | :--- | :--- |
| **DAIGT V2** | 69,520 | Core Human vs AI baseline | High-quality mix of student essays and LLM generations. Cleaned to remove duplicates. |
| **HC3 (Hello-SimpleAI)** | 22,019 | QA/Short-form diversity | Added conversational and QA patterns. Short rows (<50 words) were explicitly dropped. |
| **Navjot Kaushal (Kaggle)** | 938 | Specialized AI text | Contributed purely AI-generated text. (Note: Originally suspected to contain human ESL, but exploratory data analysis confirmed 100% AI labels). |
| **Hard Kazakh (Kaggle)** | 4,204 | Adversarial/Paraphrased | Crucial for rare classes (`ai_polished`, `ai_paraphrased`, `human_paraphrased`). |

*Note: To evaluate false positives on non-native phrasing without access to a formal ESL-labeled dataset, we generated a synthetic test set of 15 ESL-style essays using an LLM (prompted to mimic simple vocabulary and non-idiomatic phrasing). The detector falsely flagged 3 out of 15 (20%) of these essays as AI. This highlights a known fragility when evaluating non-native human writing, underscoring the necessity of using such a tool only as one signal in a broader review process.*

---

## Model Training Methodology & AI Disclosure

### Deep Learning Classifier (Signal D)
We fine-tuned `distilbert-base-uncased` as a 5-class sentence-window classifier, trained locally on a collaborator's GPU machine (Google Colab's free-tier compute constraints made full-scale training impractical there). Training used 100,335 source essays, chunked into 423,833 sentence windows. To address severe class imbalance across the five categories, we applied class-weighted cross-entropy loss, with weights capped at a 20.0x ratio to prevent gradient instability. 

Over 2 epochs, the model reached 97.6% overall (weighted) accuracy, driven almost entirely by strong performance on the two well-represented classes: 96.8% recall on human text and 99.0% recall on fully AI-generated text. **Macro-averaged F1 — which weights all five classes equally rather than by frequency — was substantially lower, at 0.566**, reflecting a sharp asymmetry in performance: `ai_paraphrased` reached 81.8% recall (supported by 22 sentence windows in validation), but `ai_polished` and `human_paraphrased` both scored 0% recall, meaning the model did not correctly identify a single example of either category in validation. 

This is consistent with a downstream finding in our combiner evaluation, where 0 of 4 held-out `ai_polished` examples were correctly flagged as AI. We attribute this to severe underrepresentation of these two classes in available training data (94 and 158 examples respectively, pre-chunking) rather than a fundamental limitation of the architecture or feature set.

### Generative AI Tool Disclosure
- **Antigravity (Claude 3.5 Sonnet)** was used extensively as an agentic pair-programmer to write Python scripts, debug data leakage (TF-IDF similarity analysis), construct the FastAPI backend, and format frontend components.
- **ChatGPT / LLMs** were used to generate the small qualitative set of ESL-style essays and to paraphrase test text for live demonstrations.

---

## Adversarial Evaluation: Confident Failures

We aggressively stress-tested our system on rare, adversarial cases (`ai_polished`). While the binary accuracy is 99%, examining where the model is *confidently wrong* reveals crucial insights into current NLP limitations:

**Failure Case 1 (AI Polished - Personal Narrative)**
> *"Before I moved to America, my father and I shared a morning ritual: drinking Pu-erh tea in my bedroom... I harbored a secret ambition..."*
- **Actual Label:** AI Polished
- **Raw Signal Output:**
```json
{
  "ai_probability": 0.0012,
  "signals": {
    "vocabulary_density": 1.431,
    "narrative_variance": 0.016,
    "stylometry": {
      "sentence_length_variance": 73.21,
      "type_token_ratio": 0.563
    },
    "classifier_probability": 0.00938,
    "paraphrase_resistance": {
      "semantic_surface_gap": 0.201
    }
  }
}
```
- **Signal Hypothesis:** Signal D (classifier_probability) scored this heavily human (0.93% AI), meaning the deep neural layer entirely missed the subtle AI polish applied later, likely anchoring on the high organic narrative variance (0.016) and structurally diverse personal anecdotes.

**Failure Case 2 (AI Polished - Baking Essay)**
> *"An authoritative voice bellows my name across the crowded school cafeteria... Among my teachers and peers, I am known simply as 'The Baker.'"*
- **Actual Label:** AI Polished
- **Raw Signal Output:**
```json
{
  "ai_probability": 0.0011,
  "signals": {
    "vocabulary_density": 1.899,
    "narrative_variance": 0.012,
    "stylometry": {
      "sentence_length_variance": 124.20,
      "type_token_ratio": 0.594
    },
    "classifier_probability": 0.0046,
    "paraphrase_resistance": {
      "semantic_surface_gap": 0.213
    }
  }
}
```
- **Signal Hypothesis:** The model was deceived by extreme structural diversity; the massive sentence length variance (124.2) strongly matches human baselines, dragging the stylometric evaluation towards 'Human' and likely causing Signal D to completely miss the AI polish (scoring it 0.46% AI).

### Identified Combiner Vulnerability
During live testing, we identified a specific limitation: when an essay is heavily machine-paraphrased (e.g., via Quillbot), the Deep Learning layer (`Signal D`) frequently catches the AI signature (scoring >99% AI). However, because the final XGBoost Combiner was trained on an imbalanced dataset lacking sufficient paraphrase examples, it actively suppresses the Neural layer's warning, dragging the final output back down to "Human" (<50%). 

*Future Work:* This is a highly fixable vulnerability. Retraining the XGBoost combiner with heavily oversampled rare-class weights (or utilizing SMOTE) would force the combiner to trust Signal D when adversarial paraphrasing is detected.
