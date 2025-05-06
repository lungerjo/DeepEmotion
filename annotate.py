import pandas as pd
import numpy as np
from pathlib import Path

# Input and output paths
input_path = Path("/home/lungerjo/scratch/DeepEmotion/annotations/soft_classification.tsv")
output_dir = Path("/home/lungerjo/scratch/DeepEmotion/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# Load input TSV
df = pd.read_csv(input_path, sep="\t")

# Extract offset and emotion columns
offset_col = "offset"
emotion_cols = [col for col in df.columns if col.startswith("e_")]
emotion_df = df[emotion_cols]

# 1. Majority consensus one-hot
majority_df = emotion_df.copy()
majority_idx = emotion_df.values.argmax(axis=1)
majority_onehot = np.zeros_like(emotion_df.values)
majority_onehot[np.arange(len(majority_onehot)), majority_idx] = 1
majority_df.loc[:, :] = majority_onehot
majority_df.insert(0, offset_col, df[offset_col])
majority_df.to_csv(output_dir / "classification_majority.tsv", sep="\t", index=False)

# 2. Heuristic clustering
cluster_dict = {
    'anger': ['e_contempt', 'e_resent', 'e_hate'],
    'sadness': ['e_remorse', 'e_disappointment', 'e_shame', 'e_sadness'],
    'fear': ['e_fear'],
    'love': ['e_love', 'e_compassion', 'e_admiration'],
    'happiness': [
        'e_hope', 'e_relief', 'e_pride',
        'e_satisfaction', 'e_gratification', 'e_gratitude', 'e_happiness'
    ],
    'none': ['e_none', 'e_gloating']
}

clustered_df = pd.DataFrame()
clustered_df["offset"] = df["offset"]

for cluster_name, members in cluster_dict.items():
    valid = [col for col in members if col in emotion_df.columns]
    clustered_df[cluster_name] = df[valid].sum(axis=1)

# Save un-thresholded clustered version
clustered_df.to_csv(output_dir / "classification_clustered.tsv", sep="\t", index=False)

# 3–7. Thresholded versions
for threshold in range(2, 7):
    clustered_thresh = clustered_df.copy()
    cluster_cols = [col for col in clustered_thresh.columns if col != "offset"]
    argmax = clustered_thresh[cluster_cols].values.argmax(axis=1)
    maxvals = clustered_thresh[cluster_cols].values.max(axis=1)

    new_data = np.zeros_like(clustered_thresh[cluster_cols].values)
    for i in range(len(new_data)):
        if maxvals[i] >= threshold:
            new_data[i, argmax[i]] = 1
        else:
            none_idx = cluster_cols.index("none")
            new_data[i, none_idx] = 1

    clustered_thresh.loc[:, cluster_cols] = new_data
    out_path = output_dir / f"classification_clustered_thresh{threshold}.tsv"
    clustered_thresh.to_csv(out_path, sep="\t", index=False)

print("All classification label variants written successfully.")
