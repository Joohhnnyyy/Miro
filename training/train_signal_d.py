import os
import json
import torch
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import classification_report

import datasets.config
datasets.config.TORCHVISION_AVAILABLE = False

# =============================================================================
# Config
# =============================================================================
LABEL_MAP = {"human": 0, "ai": 1, "ai_polished": 2, "ai_paraphrased": 3, "human_paraphrased": 4}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = list(LABEL_MAP.keys())

# If you'd rather train binary for speed, set this True — collapses the 5 classes
# down to 0=human/1=ai at load time instead of training the full taxonomy.
USE_BINARY = False

WINDOW_SENTENCES = 3          # sentence-window size for chunking long essay-level text
CHUNK_THRESHOLD_SENTENCES = 4 # rows with MORE sentences than this get chunked; shorter rows
                               # (e.g. your own sentence-diffed ai_polished/paraphrased rows) are left as-is

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
nlp.add_pipe("sentencizer")

# =============================================================================
# Step 1: Load raw processed data
# =============================================================================
def load_raw(data_dir):
    train_df = pd.read_json(os.path.join(data_dir, "train.jsonl"), lines=True)
    val_df = pd.read_json(os.path.join(data_dir, "val.jsonl"), lines=True)
    print(f"Loaded train: {len(train_df)} rows, val: {len(val_df)} rows")
    return train_df, val_df

# =============================================================================
# Step 2: Chunk long rows into sentence windows, leave short rows untouched
# =============================================================================
def chunk_dataframe(df, verbose=True):
    out_rows = []
    n_chunked = 0
    n_passthrough = 0

    # Process all text in massive batches (50x faster)
    docs = nlp.pipe(df["text"].astype(str), batch_size=1000)
    
    for row, doc in zip(df.to_dict('records'), docs):
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue

        sents = [s.text.strip() for s in doc.sents if s.text.strip()]

        if len(sents) <= CHUNK_THRESHOLD_SENTENCES:
            out_rows.append({
                "text": text,
                "prompt_topic": row.get("prompt_topic", "unknown"),
                "source": row.get("source", "unknown"),
                "label": row["label"],
            })
            n_passthrough += 1
        else:
            for i in range(0, len(sents), WINDOW_SENTENCES):
                window_text = " ".join(sents[i:i + WINDOW_SENTENCES])
                if window_text.strip():
                    out_rows.append({
                        "text": window_text,
                        "prompt_topic": row.get("prompt_topic", "unknown"),
                        "source": row.get("source", "unknown"),
                        "label": row["label"],
                    })
            n_chunked += 1

    if verbose:
        print(f"  {n_chunked} long rows chunked into windows, {n_passthrough} short rows passed through")
        print(f"  Total output rows: {len(out_rows)}")

    return pd.DataFrame(out_rows)

# =============================================================================
# Step 3: Map labels
# =============================================================================
def map_label(label_str):
    if USE_BINARY:
        return 1 if "ai" in label_str else 0
    if label_str not in LABEL_MAP:
        raise ValueError(f"Unmapped label: {label_str}")
    return LABEL_MAP[label_str]

# =============================================================================
# Step 4: Full pipeline
# =============================================================================
def train_classifier(data_dir, output_dir, model_name="distilbert-base-uncased"):
    print(f"Loading datasets from {data_dir}...")
    train_df, val_df = load_raw(data_dir)

    print("\nChunking train set...")
    train_df = chunk_dataframe(train_df)
    print("Chunking val set...")
    val_df = chunk_dataframe(val_df)

    train_df["label_id"] = train_df["label"].map(map_label)
    val_df["label_id"] = val_df["label"].map(map_label)

    print(f"\nFinal train label distribution:\n{train_df['label'].value_counts()}")
    print(f"\nFinal val label distribution:\n{val_df['label'].value_counts()}")

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]], preserve_index=False)
    dataset = DatasetDict({"train": train_ds, "val": val_ds})

    print(f"\nLoading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")
    tokenized_datasets.set_format("torch")

    print("Loading model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    num_labels = 2 if USE_BINARY else 5
    id2label = {0: "human", 1: "ai"} if USE_BINARY else ID2LABEL
    label2id = {"human": 0, "ai": 1} if USE_BINARY else LABEL_MAP

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        target_names = ["human", "ai"] if USE_BINARY else LABEL_NAMES
        report = classification_report(
            labels, preds, labels=list(range(num_labels)),
            target_names=target_names, output_dict=True, zero_division=0
        )
        metrics = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
        }
        for name in target_names:
            metrics[f"{name}_recall"] = report[name]["recall"]
        return metrics

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    class NanGuardCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                if loss is None or (isinstance(loss, float) and loss != loss):
                    print(f"!! NaN loss at step {state.global_step} — stopping.")
                    control.should_training_stop = True

    # COMPUTE CLASS WEIGHTS
    import torch.nn as nn
    class_counts = train_df["label_id"].value_counts().reindex(range(num_labels), fill_value=1).sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (num_labels * class_counts)
    
    # Cap extreme weights so rare-class batches don't dominate/destabilize training
    class_weights = np.clip(class_weights, a_min=0.1, a_max=20.0)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nComputed (capped) Class Weights: {class_weights}")

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
        callbacks=[NanGuardCallback()],
    )

    print("\nStarting training...")
    trainer.train()

    final_metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    final_model_dir = os.path.join(output_dir, "final_model")
    print(f"\nSaving best model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    with open(os.path.join(final_model_dir, "eval_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Training complete.")

if __name__ == "__main__":
    train_classifier(
        data_dir="/content/data/processed",
        output_dir="/content/training/signal_d"
    )
