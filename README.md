# AI Admissions Essay Detector


https://github.com/user-attachments/assets/78796f2e-803e-4a46-a6bd-f60788f938d7



**Live Demo:** [https://themiro.vercel.app](https://themiro.vercel.app)  
**Hugging Face Model:** [https://huggingface.co/neo2454132/miro-detector-model](https://huggingface.co/neo2454132/miro-detector-model)  

**Protecting Academic Integrity through Adversarial Resilience and Linguistic Forensics**

## Overview
Traditional AI detectors rely on basic probability mapping, making them highly vulnerable to simple paraphrase attacks—such as processing a generative essay through paraphrasing tools. This project introduces a next-generation AI forensics engine engineered to expose adversarial laundering, paraphrasing, and humanizing attacks that typically bypass conventional detectors. 

Our system achieves a highly robust **99% binary accuracy** (Human vs. AI) across out-of-domain data, while actively identifying and documenting vulnerabilities in edge-case adversarial classes. The architecture pairs a high-performance Python FastAPI backend with a modern, cinematic React frontend.

---

## Technical Architecture

### Frontend (React & TypeScript)
The user interface is built with React and TypeScript, featuring a brutalist, typography-driven aesthetic inspired by modern forensic data platforms. 
- **Cinematic Preloader:** A custom canvas-based preloader that utilizes polar coordinate mathematics for organic pixel assembly and scattering.
- **Dynamic Data Visualization:** Real-time visual feedback for classification metrics and linguistic analysis.
- **State Management & Routing:** Handled via React Router with preserved canvas states across transitions.

### Backend (FastAPI & PyTorch)
The forensic engine is powered by a high-throughput FastAPI service, orchestrating a multi-layered classification pipeline.
- **Local Inference:** Model inference is executed locally using PyTorch, ensuring data privacy and low-latency processing without external API dependencies.
- **The 5-Signal Pipeline:** Each text submission is analyzed across five independent dimensions before a final XGBoost Combiner executes the ultimate classification.

---

## The 5-Signal Pipeline

1. **Deep Learning Classification (Signal D)**: *The Neural Engine.* We utilized a fine-tuned `DeBERTa` model, analyzing deep, high-dimensional signatures unique to Large Language Models.
2. **Paraphrase Resistance Analysis**: *The Launder-Catcher.* Measures Semantic Surface Gap and Rhythm Autocorrelation. Paraphrasing destroys natural sentence flow; this signal catches the mechanical choppiness left behind.
3. **Narrative Variance Scoring**: *The Predictability Check.* Human writers naturally vary sentence lengths and structural cadence. This signal measures the mathematical predictability of the essay's narrative flow.
4. **Forensic Stylometry**: *The Grammar Fingerprint.* Extracts hard linguistic features (Type-Token Ratio, passive voice frequency, punctuation diversity) to compare against known human baselines.
5. **AI Vocabulary Density**: *The Tell-Tale Lexicon.* Scans for statistically anomalous concentrations of known generative transitional phrases.

---

## Dataset Provenance

| Source / Dataset | Total Rows | Contribution | Notes & Dropped Data |
| :--- | :--- | :--- | :--- |
| **DAIGT V2** | 69,520 | Core Human vs AI baseline | High-quality mix of student essays and LLM generations. Cleaned to remove duplicates. |
| **HC3 (Hello-SimpleAI)** | 22,019 | QA/Short-form diversity | Added conversational and QA patterns. Short rows (<50 words) explicitly dropped. |
| **Navjot Kaushal (Kaggle)** | 938 | Specialized AI text | Contributed purely AI-generated text. Exploratory data analysis confirmed 100% AI labels. |
| **Hard Kazakh (Kaggle)** | 4,204 | Adversarial/Paraphrased | Crucial for rare classes (`ai_polished`, `ai_paraphrased`, `human_paraphrased`). |

*Note: To evaluate false positives on non-native phrasing without access to a formal ESL-labeled dataset, a synthetic test set of 15 ESL-style essays was generated. The detector flagged 3 out of 15 (20%) of these essays as AI. This highlights a known fragility when evaluating non-native human writing, underscoring the necessity of using such a tool as one signal in a broader review process.*

---

## Model Training Methodology

### Deep Learning Classifier (Signal D)
We fine-tuned a base transformer model as a 5-class sentence-window classifier. Training utilized 100,335 source essays, chunked into 423,833 sentence windows. To address severe class imbalance across the categories, class-weighted cross-entropy loss was applied, with weights capped at a 20.0x ratio to prevent gradient instability. 

Over 2 epochs, the model reached 97.6% overall (weighted) accuracy, driven by strong performance on the two well-represented classes: 96.8% recall on human text and 99.0% recall on fully AI-generated text. **Macro-averaged F1 was substantially lower at 0.566**, reflecting a sharp asymmetry in performance on adversarial classes (e.g., `ai_polished` and `human_paraphrased`). This is attributed to severe underrepresentation of these classes in available training data.

---

## Adversarial Evaluation: Confident Failures

We aggressively stress-tested our system on rare, adversarial cases. While binary accuracy is 99%, examining where the model is *confidently wrong* reveals crucial insights into current NLP limitations:

**Failure Case 1 (AI Polished - Personal Narrative)**
> *"Before I moved to America, my father and I shared a morning ritual: drinking Pu-erh tea in my bedroom... I harbored a secret ambition..."*
- **Actual Label:** AI Polished
- **Signal Hypothesis:** Signal D scored this heavily human (0.93% AI), meaning the deep neural layer entirely missed the subtle AI polish applied later, anchoring on the high organic narrative variance (0.016) and structurally diverse personal anecdotes.

**Failure Case 2 (AI Polished - Baking Essay)**
> *"An authoritative voice bellows my name across the crowded school cafeteria... Among my teachers and peers, I am known simply as 'The Baker.'"*
- **Actual Label:** AI Polished
- **Signal Hypothesis:** The model was deceived by extreme structural diversity; the massive sentence length variance (124.2) strongly matches human baselines, causing Signal D to completely miss the AI polish (scoring it 0.46% AI).

### Identified Combiner Vulnerability
During live testing, we identified a specific limitation: when an essay is heavily machine-paraphrased, the Deep Learning layer (`Signal D`) frequently catches the AI signature (scoring >99% AI). However, because the final XGBoost Combiner was trained on an imbalanced dataset lacking sufficient paraphrase examples, it actively suppresses the Neural layer's warning, dragging the final output back down to "Human" (<50%). 

*Future Work:* Retraining the XGBoost combiner with heavily oversampled rare-class weights (or utilizing SMOTE) will force the combiner to trust Signal D when adversarial paraphrasing is detected.
