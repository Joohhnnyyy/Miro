# MASTER BUILD PROMPT v3 — AI Admissions Essay Detector (Hardened Restart)

Paste this whole document into Antigravity. This restarts the build from zero, with three rounds
of debugging lessons folded in as explicit rules — not just fixes applied once, but constraints
the agent should follow proactively so the same bug classes don't recur.

---

## 0. Lessons baked into this version — read before building

Three real bugs surfaced in earlier attempts. Each one below produced a specific rule that now
applies throughout this document, not just at the place it was first found.

1. **Silent "pick the first file" bugs.** A loader using `glob.glob(...)[0]` or similar silently
   dropped whole files (a `val.jsonl` alongside `train.jsonl`; other files in a multi-file Kaggle
   download) with no error — the pipeline ran clean, just on less data than it should have.
   **Rule: any file-loading step must load and concatenate every matching file it finds, print how
   many files and rows it loaded, and never index into a glob result.**

2. **Silent placeholder overwrite of real metadata.** An early normalization pass hardcoded
   `"source": "training_data_zip"` and `"prompt_topic": "unknown"` even when the raw data already
   had real values for both. This quietly destroyed the provenance tracking the whole
   topic-concentration and per-source eval design depends on. **Rule: never hardcode a metadata
   field if the raw source already has real data for it — check and use the real column first.**

3. **DeBERTa-v3 + `gradient_checkpointing=True` (and, separately, too-high a learning rate) caused
   a NaN loss collapse on the very first training step** — loss frozen at `0.000000`, validation
   loss `nan`, model collapsing to always predicting the majority class. Identical metrics across
   all epochs is the signature of this failure. **Rule: train with `gradient_checkpointing=False`,
   `learning_rate=1e-5`, `fp16=False`, `max_grad_norm=1.0`, and a NaN-guard callback that stops
   training immediately if it recurs — check the loss value at step ~50 before letting a run go
   the full distance. If it still collapses, switch to `roberta-base` rather than continuing to
   debug DeBERTa's config.**

One more standing rule this version adds up front, since it would have caught bug #1 and #3 much
earlier: **process one data source at a time, in isolation, print a sample row and label
distribution, and visually confirm it before moving to the next source or merging anything.**
Slower, but it turns a silent multi-step failure into an immediate, visible one.

---

## 1. What we are building

A deployed web app. User pastes a college admissions essay. The app returns:
- Sentence/paragraph-level highlighting (color-coded by AI-likelihood)
- A "why" explanation per flagged sentence (which signals fired)
- An aggregate breakdown by section — never a single flat percentage
- An honest evaluation page: real accuracy, 3 confident failures, an ESL false-positive check, and
  a paraphrase-attack degradation check

**Hard constraint, non-negotiable:** no chat-completion model may issue the human/AI verdict.
LLMs and paraphrase models are instruments only — generating synthetic training data, producing
embeddings, or narrating an already-computed score after the fact. Never the judge. A detector
that sends the essay to a chat model and asks for a verdict is a wrapper, is unreliable, can't
explain itself, and will be recognized instantly.

---

## 2. Detection logic — five signals combined

### Signal A — Vocabulary Signature
AI models converge on a narrow, "statistically safe" vocabulary cluster and repeat it across
documents; humans vary diction unpredictably based on context and tone.

1. Compute word/phrase frequency separately for human-labeled and AI-labeled training text.
2. Compute a Dirichlet-smoothed log-odds ratio per n-gram (unigram + bigram), n-grams appearing
   ≥5 times only. **Do not strip stopwords before forming n-grams** — many real AI tells
   ("in conclusion," "it is important to note") are built entirely from stopwords, and stripping
   them first silently deletes these phrases before they're ever counted.
3. Convert each log-odds value to a **z-score** (`delta / sqrt(1/count_ai + 1/count_human)`, per
   Monroe et al.) and keep only words above a significance threshold (e.g. `z >= 1.96`) — raw
   log-odds alone over-ranks rare, noisy words.
4. **Gate on topic concentration**: for each candidate phrase, compute what fraction of its total
   occurrences come from a single dominant `prompt_topic`. Drop phrases whose concentration
   exceeds ~0.6 — this is what catches topic leakage (e.g. "car usage" scoring as an AI signature
   purely because one dataset's AI examples happened to cluster around a "limiting car usage"
   prompt). Requires real `prompt_topic` metadata per row — see Section 0, lesson 2.
5. Apply a manual exclusion list for anything that still looks topic-specific after the above.
6. Keep the top ~500 surviving n-grams, ranked by z-score. Save as `signature_list.json`:
   `{ "phrase": ..., "log_odds": ..., "z_score": ..., "topic_concentration": ..., "direction": "ai" }`.
   Also save `excluded_by_concentration.json` — the list of what got filtered and why; this is
   direct evidence for the eval write-up.
7. At inference: score a sentence by density of matched signature phrases weighted by z-score;
   require a minimum density before it counts — one flagged word alone is not a signal.

### Signal B — Narrative Consistency / Direction Drift
AI text stays too consistent in direction/tone; humans introduce tangents and tonal shifts.

1. Split the essay into sentences (or 2-sentence windows).
2. Embed each segment with `sentence-transformers/all-MiniLM-L6-v2` (small, local, CPU-fine).
3. Compute cosine similarity between consecutive segment embeddings.
4. Compute the **variance** of that similarity sequence across the essay (variance matters more
   than the mean). Low variance → AI-like. High variance → human-like.
5. This is embedding-space geometry, not perplexity — keep it distinct from Signal D.

### Signal C — Stylometric Features
Pure text computation, no model: sentence length variance, type-token ratio, transition-phrase
density, passive voice ratio, punctuation diversity, readability score. `spaCy` (`en_core_web_sm`)
for POS/passive voice, `textstat` for readability.

### Signal D — Fine-Tuned Classifier
Fine-tune `microsoft/deberta-v3-small` as a **5-class sentence-level classifier** (see Section 3
for labels). Fallback: `roberta-base` — same `AutoTokenizer`/`AutoModelForSequenceClassification`
API, drop-in swap, use it immediately if DeBERTa collapses twice (Section 0, lesson 3).

*Training config — hardened per lesson 3 above. Pick the hardware block that matches your setup:*

```python
# --- Colab T4 (CUDA, 16GB VRAM) ---
TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    fp16=False,                       # DeBERTa-v3 + fp16 is a known NaN-collapse combo — leave off
    gradient_checkpointing=False,     # known incompatibility with DeBERTa-v2/v3 attention — leave off
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    learning_rate=1e-5,               # lower than the typical 2e-5 default — DeBERTa-v3 needs this
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",
)
```

```python
# --- Apple Silicon (M1/M2/M3/M4), MPS backend ---
TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=False,
    bf16=False,
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    learning_rate=1e-5,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",
    use_mps_device=True,
)
```

Add a NaN-guard callback to every training run, regardless of hardware:

```python
from transformers import TrainerCallback

class NanGuardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if loss is None or (isinstance(loss, float) and loss != loss):  # NaN check
                print(f"!! NaN loss at step {state.global_step} — stopping.")
                control.should_training_stop = True
```

Check the loss value around step 50 before letting a run continue — it should be a real,
decreasing number (starting near `ln(5) ≈ 1.6` for 5-class), not `0.000000` or `nan`.

### Signal E — Paraphrase-Resistant Features
Paraphrasers preserve meaning while flattening structure even further than the original AI draft.
Signal E targets the *gap* between meaning and surface form, which survives paraphrasing better
than raw perplexity or burstiness alone.

**E1 — Semantic-surface divergence**: high semantic similarity to neighboring sentences combined
with low surface n-gram overlap with those same neighbors.
```python
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_surface_gap(sentences):
    embs = embedder.encode(sentences)
    gaps = []
    for i in range(1, len(sentences) - 1):
        sem_sim = np.dot(embs[i], embs[i-1]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[i-1]))
        toks_a, toks_b = set(sentences[i].lower().split()), set(sentences[i-1].lower().split())
        surface_overlap = len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)
        gaps.append(sem_sim - surface_overlap)
    return gaps
```

**E2 — Synonym-substitution density** (rough heuristic, weight lightly, document the limitation):
```python
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)

def synonym_substitution_score(doc):  # spaCy doc
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
    return sum(scores) / len(scores) if scores else 0.0
```

**E3 — Residual rhythm autocorrelation** (lag-1 autocorrelation of sentence length, survives
sentence-by-sentence rewriting better than raw variance):
```python
def rhythm_autocorrelation(sentence_lengths):
    if len(sentence_lengths) < 3:
        return 0.0
    arr = np.array(sentence_lengths, dtype=float) - np.mean(sentence_lengths)
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if arr.std() > 0 else 0.0
```

### Combining all five signals
Train `sklearn.linear_model.LogisticRegression` on [Signal A, Signal B variance, Signal C vector,
Signal D probabilities, Signal E's three features] → final per-sentence score. Log the learned
coefficients — this is the "how and why" the brief asks you to defend. Save `combiner_model.pkl`.
If Signal E's weights end up near zero, report that plainly — a legitimate finding, not a failure.

*Permitted LLM/paraphraser roles, and only these:* (1) generating synthetic AI essays for training
data, (2) a dedicated paraphrase model generating paraphrase-attack training data, (3) optionally,
after the score is already computed, narrating the evidence in plain English for the UI. Never
answering "is this AI?" directly.

---

## 3. Label taxonomy

| Label | Meaning |
|---|---|
| `human` | Untouched human writing |
| `ai` | Fully AI-generated |
| `ai_polished` | Human draft, AI-edited (sentence-diffed against the original) |
| `ai_paraphrased` | AI-generated, then paraphrased |
| `human_paraphrased` | Human-written, then paraphrased — **critical negative control** |

`human_paraphrased` prevents the classifier from learning "paraphrase-flat prose = AI," which
would reproduce the ESL-bias failure mode the brief explicitly warns about. Train 5-class, collapse
`{ai, ai_polished, ai_paraphrased}` → positive and `{human, human_paraphrased}` → negative for the
UI's final score, but keep the 5-class breakdown for per-sentence explanations and eval reporting.

**Reality check on sourcing:** most public Kaggle datasets are `human`/`ai` binary only — the three
richer categories almost entirely come from your own generation work (Section 4.4). Expect your
combined dataset to be effectively binary until that step is done; `0.0` recall on the other three
classes before then is expected, not a bug.

---

## 4. Datasets

Document sources, counts, and coverage gaps as you go — this becomes eval-section evidence later.

### 4.1 Existing labeled corpora
```
https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

https://huggingface.co/datasets/Hello-SimpleAI/HC3

https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays

https://www.kaggle.com/datasets/hardkazakh/ai-generated-vs-human-written-text-dataset

https://www.kaggle.com/datasets/lburleigh/tla-lab-ai-detection-for-essays-aide-dataset

```
Download via `kagglehub` (needs a free Kaggle account + `kaggle.json` API token):
```python
import kagglehub
path = kagglehub.dataset_download("thedrcat/daigt-v2-train-dataset")
print(path)
```
**Immediately after any download, list every file it actually produced — do not assume a single
file.** Some of these datasets ship as one CSV; others as multiple. `daigt_v2` specifically ships
as a separate `train.jsonl` + `val.jsonl` — a loader that only reads one silently trains on less
data than exists (this happened once already; see Section 0).
```python
import os
for root, dirs, files in os.walk(path):
    for f in files:
        print(os.path.join(root, f))
```

The **AIDE dataset is essay-specific and closest in domain to this task** — weight it more heavily
or tag it distinctly for later per-source accuracy checks.

### 4.2 Multi-Language Evaluation Dataset (local zip, already downloaded)
Treat as **evaluation-only by default** — multilingual text will confuse the English-tuned
classifier and Signals A/C. Inspect structure first:
```bash
unzip -l "AI-Detector Multi-Language Evaluation Dataset.zip"
```
Hold out the full multilingual set as a documented limitation in the eval report, rather than
mixing it into training.

### 4.3 Scraped real admissions-essay sources
```
https://www.collegeessayguy.com/blog/college-essay-examples
https://apply.jhu.edu/hopkins-insider/the-secret-ingredient-is-connection/
```
Real, provenance-known human essays. Use `requests` + `BeautifulSoup`, inspect each page's actual
HTML before writing selectors, tag `source` distinctly. Manual copy-paste into a spreadsheet
(`text, prompt_topic, source` columns) is an acceptable fallback if scraping eats too much time —
100-150 essays by hand is a safer time investment than debugging scrapers across six sites.

Target 150-300+ human essays total, spread across multiple Common App prompt topics (identity,
setback, challenging a belief, gratitude, growth, curiosity, community) — not just one or two
topics. A detector trained on one topic distribution performs unpredictably on others, and skewed
topic coverage is exactly what caused the Signal A leakage bug in Section 2.

### 4.4 Your own generated data
**AI-generated essays** — 2-3 different models, same prompts as the human essays, varying
temperature (0.5/0.7/1.0), so the classifier learns "AI-ness" broadly rather than one model's
fingerprint. Target 50-100 essays.

**AI-polished essays** — 30-50 human essays, "improve the flow and grammar" prompt, then
sentence-diff against the original with `difflib.SequenceMatcher` so you know exactly which
sentences were touched (`equal` → still `human`, anything else → `ai_polished`).

**Paraphrase-attack data** — a dedicated paraphrase model (e.g. `tuner007/pegasus_paraphrase`),
not a general chat model, run sentence-by-sentence over both existing AI essays (→
`ai_paraphrased`) and a subset of human essays (→ `human_paraphrased`). Keep sentence-level
alignment so Signal E's neighbor-comparison features can be validated. Cache results — this is
slow to regenerate.

### 4.5 Process each source in isolation, then merge

**Per lesson from Section 0: one source at a time, verify, then move on.**

```python
import glob, pandas as pd

def load_all_files(directory, verbose=True):
    """Loads and concatenates every csv/json/jsonl file under directory. Never picks just one."""
    csv_files = glob.glob(f"{directory}/**/*.csv", recursive=True)
    json_files = glob.glob(f"{directory}/**/*.json*", recursive=True)
    frames = []
    for f in csv_files + json_files:
        df = pd.read_csv(f) if f.endswith(".csv") else pd.read_json(f, lines=f.endswith(".jsonl"))
        df["_source_file"] = f
        frames.append(df)
        if verbose:
            print(f"  loaded {f}: {len(df)} rows, columns={df.columns.tolist()}")
    if not frames:
        raise FileNotFoundError(f"No csv/json files found under {directory}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"TOTAL from {directory}: {len(combined)} rows across {len(frames)} file(s)\n")
    return combined

def process_source(directory, source_name, text_col, label_col, topic_col,
                    raw_label_map, output_path):
    print(f"=== Processing {source_name} ===")
    raw_df = load_all_files(directory)

    normalized_df = pd.DataFrame({
        "text": raw_df[text_col],
        "prompt_topic": raw_df[topic_col] if topic_col and topic_col in raw_df.columns else "unknown",
        "source": source_name,
        "label": raw_df[label_col].map(raw_label_map),
    })

    missing = normalized_df["label"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows failed label mapping.")
        print(f"  Raw unique label values: {raw_df[label_col].unique()}")
        raise ValueError(f"{source_name}: fix raw_label_map before proceeding.")

    normalized_df.to_json(output_path, orient="records", lines=True)
    print(f"  Saved {len(normalized_df)} rows -> {output_path}")
    print(f"  Label distribution:\n{normalized_df['label'].value_counts()}\n")
    print(f"  Sample row:\n{normalized_df.iloc[0].to_dict()}\n")
    return normalized_df
```

Run once per source, inspecting the printed output each time before proceeding to the next:
```python
daigt_df = process_source(
    directory="data/raw/daigt_v2", source_name="daigt_v2",
    text_col="text", label_col="label", topic_col="prompt_topic",
    raw_label_map={"human": "human", "ai": "ai"},
    output_path="data/processed/daigt_v2_normalized.jsonl",
)
```
Repeat for each remaining source (`navjotkaushal`, `hardkazakh`, `aide`, `ayeshaseherr`, HC3, the
scraped essays, and the three generated categories), filling in each source's real column names
and raw label values from what `load_all_files` prints — never assume a schema without checking.

### 4.6 Merge and split — only after every source is individually verified
```python
frames = [pd.read_json(f, lines=True) for f in glob.glob("data/processed/*_normalized.jsonl")]
combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")
print("Sources merged:", glob.glob("data/processed/*_normalized.jsonl"))
print(combined.groupby(["source", "label"]).size())
combined.to_json("data/processed/combined.jsonl", orient="records", lines=True)

from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(combined, test_size=0.2, stratify=combined["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

train_df.to_json("data/processed/train.jsonl", orient="records", lines=True)
val_df.to_json("data/processed/val.jsonl", orient="records", lines=True)
test_df.to_json("data/processed/test.jsonl", orient="records", lines=True)
print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
```
Split by document, never by sentence (leaks context), stratified by `label`.

---

## 5. Repo structure

```
ai-essay-detector/
├── data/
│   ├── raw/                          # per-source subfolders, untouched downloads
│   ├── generated/                    # own AI / polished / paraphrased essays
│   └── processed/
│       ├── <source>_normalized.jsonl # one per source, verified individually
│       ├── combined.jsonl
│       ├── train.jsonl / val.jsonl / test.jsonl
├── training/
│   ├── build_signature_list.py       # Signal A, with topic-concentration gate
│   ├── generate_paraphrases.py       # Signal E data
│   ├── train_classifier.py           # Signal D, 5-class, hardened config
│   └── train_combiner.py             # final logistic regression, signals A-E
├── backend/
│   ├── main.py
│   ├── signals/
│   │   ├── vocabulary.py / narrative.py / stylometry.py / classifier.py / paraphrase_resistance.py
│   ├── combine.py
│   └── model_weights/
├── frontend/                         # React + Vite + Tailwind
├── eval/
│   ├── evaluate.py                   # per-class recall, 3 confident failures
│   ├── esl_bias_check.py
│   └── paraphrase_attack_eval.py
└── README.md
```

---

## 6. Execution order

1. **Data — one source at a time.** Download/locate each source, run `load_all_files` +
   `process_source` individually, inspect the printed sample row and label distribution before
   moving to the next. Only merge (Section 4.6) once every source is verified.
2. **Signal A:** log-odds + z-score + topic-concentration gate. Save `signature_list.json` and
   `excluded_by_concentration.json`. Manually skim top 30-40 survivors for leftover leakage.
3. **Signal B + C:** pure code, sanity-check on a couple of sample essays before trusting them.
4. **Signal E:** implement, unit-test on sample sentence pairs from the paraphrase data — confirm
   the semantic-surface gap is actually higher on paraphrased text before wiring into the combiner.
5. **Signal D:** fine-tune with the hardened config for your hardware (Section 2). Watch the loss
   at step ~50 before letting the run continue. Switch to `roberta-base` immediately if it
   collapses twice.
6. **Combiner:** run all five signals over the train split, fit logistic regression, log
   coefficients.
7. **Backend:** FastAPI, load all artifacts at startup, test locally before the frontend.
8. **Frontend:** paste-and-highlight UI, per-sentence tooltip showing which of the 5 classes drove
   the score and why.
9. **Eval:** held-out test set, per-class recall (especially the two paraphrase classes), 3
   confident failures, ESL bias check, adversarial paraphrase-attack degradation check. Save
   `eval_report.json`, surface it in an "About this detector" page.
10. **Deploy:** backend → Hugging Face Spaces (Docker), frontend → Vercel.
11. **Docs:** README with architecture, key decisions, honest results, dataset provenance and
    coverage gaps, AI-tool-usage disclosure.

---

## 7. Instruction to the coding agent (Antigravity)

Build this project exactly as specified above, from zero, executing the full pipeline in the order
given in Section 6. **Process each data source individually and print its sample row and label
distribution before moving to the next or merging anything — do not write one combined ingestion
script that touches multiple sources at once.** Never index into a file glob (`[0]` or similar);
always load and concatenate every matching file in a directory, and print how many files and rows
were loaded. Never hardcode a metadata field (`source`, `prompt_topic`) if the raw data already
contains a real value for it. For Signal D training, use the hardened config in Section 2 for the
detected hardware (CUDA vs MPS), include the NaN-guard callback, and check the loss value around
step 50 before continuing a run — if `roberta-base` becomes necessary after two DeBERTa collapses,
switch immediately rather than continuing to tune DeBERTa's hyperparameters. Use PyTorch +
HuggingFace Transformers for Signal D, sentence-transformers for Signal B and E, spaCy + textstat +
nltk/WordNet for Signal C and E, a dedicated Pegasus paraphrase model for paraphrase-attack data
generation, scikit-learn/joblib for Signal A and the combiner, FastAPI for the backend, React +
Vite + Tailwind for the frontend. Do not call any external chat-completion model to produce the
human/AI verdict directly — chat models and the paraphrase model are only for synthetic data
generation and post-hoc narration of an already-computed score. Ask for confirmation before moving
to the next numbered step in Section 6, and print dataset sizes, per-source label balance, and
per-class eval metrics at each stage so I can verify before proceeding.