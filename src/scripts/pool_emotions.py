import os
import pandas as pd
from collections import Counter
from functools import reduce

# Set the directory path where your updated TSV files are located
tsv_dir = "/home/paperspace/DeepEmotion/data/updated_annotations"

# List all .tsv files in the directory
tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]

dfs = []
for idx, tsv_file in enumerate(sorted(tsv_files)):
    file_path = os.path.join(tsv_dir, tsv_file)
    
    df = pd.read_csv(file_path, sep="\t")
    # Rename the 'emotion' column to avoid collisions
    df = df.rename(columns={"emotion": f"emotion_{idx}"})
    
    # Rename the first column to 'step' if it's not already
    if df.columns[0] != "step":
        df = df.rename(columns={df.columns[0]: "step"})
    
    # Keep only 'step' + the new emotion column
    dfs.append(df[["step", f"emotion_{idx}"]])

# Merge all DataFrames on 'step'
df_merged = reduce(lambda left, right: pd.merge(left, right, on="step", how="outer"), dfs)

def pooled_label_and_count(row):
    """
    For each row, examine all emotion_* columns:
      1) If all are NONE (or empty), pooled_emotion = "NONE" and num_judges = 0.
      2) Otherwise, pick the label with the highest frequency if it appears at least twice.
         If no label appears at least twice, return "NONE", 0.
    """
    emotion_cols = [col for col in row.index if col.startswith("emotion_")]
    
    # Gather all labels except 'NONE'
    labels = [row[col] for col in emotion_cols if row[col] != 'NONE']
    
    if len(labels) == 0:
        # All judges said NONE (or no labels)
        return "NONE", 0
    
    # Count label occurrences
    counts = Counter(labels)
    most_common_label, max_count = counts.most_common(1)[0]
    
    # Ensure the label appears at least twice
    if max_count < 5:
        return "NONE", 0
    
    return most_common_label, max_count

# Apply the function to each row
df_merged[["pooled_emotion", "num_judges"]] = df_merged.apply(
    lambda row: pd.Series(pooled_label_and_count(row)), axis=1
)

# Now rename columns to the desired structure:
# 1) 'step' -> 'offset'
# 2) 'pooled_emotion' -> 'emotion'
# 3) 'emotion_{idx}' columns -> 'judge_{idx}'

df_merged.rename(
    columns={"step": "offset", "pooled_emotion": "emotion"},
    inplace=True
)

# Rename indexed emotion columns to judge_{idx}
for col in df_merged.columns:
    if col.startswith("emotion_"):
        new_col = col.replace("emotion_", "judge_")
        df_merged.rename(columns={col: new_col}, inplace=True)

# Reorder columns so the first column is 'offset',
# the second is 'emotion',
# followed by judge_* columns, and finally num_judges.
judge_columns = [c for c in df_merged.columns if c.startswith("judge_")]
other_columns = [c for c in df_merged.columns if c not in ["offset", "emotion"] + judge_columns]
df_merged = df_merged[["offset", "emotion"] + judge_columns + other_columns]

# Preview the resulting DataFrame
print(df_merged.head(10))

# Export the final DataFrame as a TSV
output_path = os.path.join(tsv_dir, "pooled_annotations_structured.tsv")
df_merged.to_csv(output_path, sep="\t", index=False)

print(f"\nExported structured TSV to: {output_path}")

# Count and print the frequency of each emotion
emotion_counts = Counter(df_merged["emotion"])
print("\nFrequency of pooled_emotion labels:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count}")
