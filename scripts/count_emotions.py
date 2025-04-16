import pandas as pd

tsv_path = "/home/paperspace/DeepEmotion/data/resampled_annotations/av1o6_resampled.tsv"

import pandas as pd

def count_emotions():
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Check if 'emotion' column exists
    if 'emotion' not in df.columns:
        raise ValueError("The 'emotion' column is not present in the TSV file.")
    
    # Count the occurrences of each unique string in the 'emotion' column
    emotion_counts = df['emotion'].value_counts()
    
    # Convert the result to a dictionary for easier use
    emotion_dict = emotion_counts.to_dict()
    
    # Count rows where 'emotion' is not 'NONE'
    non_none_count = (df['emotion'] != 'NONE').sum()
    
    return emotion_dict, non_none_count

if __name__ == "__main__":
    emotion_dict, non_none_count = count_emotions()
    print(f"emotion_dict {emotion_dict}")
    print(f"non_none_count {non_none_count}")