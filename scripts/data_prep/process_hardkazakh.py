import glob
import pandas as pd
import os

def load_all_files(directory, verbose=True):
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

if __name__ == "__main__":
    import pandas as pd
    import json
    
    # Process AI
    ai_df = pd.read_csv("data/raw/hardkazakh/AI_Generated.csv")
    ai_rows = []
    for _, row in ai_df.iterrows():
        ai_rows.append({
            "text": row["Generated"],
            "prompt_topic": row.get("Topic", "unknown"),
            "source": "hardkazakh",
            "label": "ai"
        })
        
    # Process Human
    human_df = pd.read_csv("data/raw/hardkazakh/Human-Written.csv")
    human_rows = []
    for _, row in human_df.iterrows():
        human_rows.append({
            "text": row["Text"],
            "prompt_topic": row.get("Topic", "unknown"),
            "source": "hardkazakh",
            "label": "human"
        })
        
    combined = pd.DataFrame(ai_rows + human_rows)
    out_path = "data/processed/hardkazakh_normalized.jsonl"
    combined.to_json(out_path, orient="records", lines=True)
    
    print(f"=== Processing hardkazakh ===")
    print(f"  Saved {len(combined)} rows -> {out_path}")
    print(f"  Label distribution:\n{combined['label'].value_counts()}\n")
    print(f"  Sample row:\n{combined.iloc[0].to_dict()}\n")
