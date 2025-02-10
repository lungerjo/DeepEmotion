import pandas as pd
import os

# Directory containing the TSV files
directory = '../data/resampled_annotations'

# Dictionary to hold the DataFrames
sub_dataframes = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('ed.tsv'):
        # Load the TSV file into a DataFrame
        df = pd.read_csv(os.path.join(directory, filename), sep='\t')
        # Store the DataFrame in the dictionary with the filename (without extension) as the key
        sub_dataframes[os.path.splitext(filename)[0]] = df

# Display the keys of the dictionary to confirm loading
print(sub_dataframes.keys())

cluster_dict = {'e_anger/rage': ['e_contempt', 'e_resent'], 'e_sadness': ['e_remorse', 'e_dissapointment', 'e_shame'], 'e_fear': ['e_fears_confirmed'], 'e_love': ['e_love'], 'e_admiration': ['e_gratitude'], 'e_happiness': ['e_hope', 'e_happy-for', 'e_relief', 'e_pride', 'e_satisfaction', 'e_gratification'], 'e_pity/compassion':['e_pity/compassion'], 'e_gloating': ['e_gloating'], 'e_hate': ['e_hate']}
# Update the cluster_dict to remove 'e_' and fully capitalize every word
updated_cluster_dict = {emotion.replace('e_', '').replace('_', ' ').upper(): [e.replace('e_', '').replace('_', ' ').upper() for e in emotions] for emotion, emotions in cluster_dict.items()}
# Remove '/' and spaces from the keys and values in the updated_cluster_dict
final_cluster_dict = {emotion.replace('/', '').replace(' ', ''): [e.replace('/', '').replace(' ', '') for e in emotions] for emotion, emotions in updated_cluster_dict.items()}
# Print the updated dictionary
print(final_cluster_dict)

# Iterate over each dataframe in sub_dataframes
for key, df in sub_dataframes.items():
    # Replace the emotion with the key of the emotion cluster it belongs to
    df['emotion'] = df['emotion'].apply(lambda x: next((cluster for cluster, emotions in final_cluster_dict.items() if x.replace('e_', '').replace('_', ' ').upper() in emotions), x))

# Display the updated dataframes to confirm the changes
for key, df in sub_dataframes.items():
    print(f"Updated DataFrame: {key}")
    print(df.head())

# Directory to save the new TSV files
output_directory = '../data/updated_annotations'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over each dataframe in sub_dataframes
for key, df in sub_dataframes.items():
    # Define the output file path
    output_file_path = os.path.join(output_directory, f'{key}_updated.tsv')
    
    # Save the dataframe as a TSV file
    df.to_csv(output_file_path, sep='\t', index=False)

# Confirm the files have been saved
print(f"Updated TSV files have been saved to {output_directory}")