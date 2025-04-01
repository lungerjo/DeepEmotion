import pandas as pd
import os
from collections import Counter

directory = '/home/paperspace/DeepEmotion/data/annotations/src/emotions/data/resampled'

def print_emotion_frequencies(df, name):
    emotion_counts = Counter(df['emotion'])
    print(f"\nEmotion frequencies for {name}:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")

sub_dataframes = {}
print("Loading TSV files...")
for filename in os.listdir(directory):
    if filename.endswith('ed.tsv'):
        df = pd.read_csv(os.path.join(directory, filename), sep='\t')
        sub_dataframes[os.path.splitext(filename)[0]] = df
        print(f"  Loaded {filename} with {len(df)} rows")

print(sub_dataframes.keys())
for key, df in sub_dataframes.items():
    print_emotion_frequencies(df, f"{key} (Before Clustering)")

# cluster_dict = {'e_anger/rage': ['e_contempt', 'e_resent'], 'e_sadness': ['e_remorse', 'e_disappointment', 'e_shame'], 'e_fear': ['e_fears_confirmed'], 'e_love': ['e_love'], 'e_admiration': ['e_gratitude'], 'e_happiness': ['e_hope', 'e_happy-for', 'e_relief', 'e_pride', 'e_satisfaction', 'e_gratification'], 'e_pity/compassion':['e_pity/compassion'], 'e_gloating': ['e_gloating']}

cluster_dict = {'e_anger': ['e_contempt', 'e_resent', 'e_angerrage', 'e_hate', 'e_resentment'], 'e_sadness': ['e_remorse', 'e_disappointment', 'e_shame'], 'e_fear': ['e_fears_confirmed'], 'e_love': ['e_love', 'e_compassion', 'admiration'], 'e_happiness': ['e_hope', 'e_happyfor', 'e_relief', 'e_pride', 'e_satisfaction', 'e_gratification', 'e_gratitude'], "NONE": ['e_gloating']}
updated_cluster_dict = {emotion.replace('e_', '').replace('_', ' ').upper(): [e.replace('e_', '').replace('_', ' ').upper() for e in emotions] for emotion, emotions in cluster_dict.items()}
final_cluster_dict = {emotion.replace('/', '').replace(' ', ''): [e.replace('/', '').replace(' ', '') for e in emotions] for emotion, emotions in updated_cluster_dict.items()}
print(final_cluster_dict)

for key, df in sub_dataframes.items():
    df['emotion'] = df['emotion'].apply(lambda x: next((cluster for cluster, emotions in final_cluster_dict.items() if x.replace('e_', '').replace('_', ' ').upper() in emotions), x))

for key, df in sub_dataframes.items():
    print_emotion_frequencies(df, f"{key} (After Clustering)")

for key, df in sub_dataframes.items():
    print(f"Updated DataFrame: {key}")
    print(df.head())

output_directory = '../data/updated_annotations'
os.makedirs(output_directory, exist_ok=True)

for key, df in sub_dataframes.items():
    output_file_path = os.path.join(output_directory, f'{key}_updated.tsv')
    df.to_csv(output_file_path, sep='\t', index=False)
    print(f"  Saved updated file: {output_file_path}")

print(f"Updated TSV files have been saved to {output_directory}")
