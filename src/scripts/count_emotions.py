import pandas as pd

tsv_path = "/Users/joshualunger/DeepEmotionBackup/data/resampled_annotations/av1o2_resampled.tsv"

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
    
    return emotion_dict

if __name__ == "__main__":
    emotion_dict = count_emotions()
    print(emotion_dict)