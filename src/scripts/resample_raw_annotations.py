import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

observer_index = 8 # which observer annotations to use

rel_file_path = rel_file_path = f"data/annotations/src/emotions/data/raw/av1o0{observer_index}.csv"
unit = 2
end_time = 7031

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def resample_events_prioritize(cfg: DictConfig):
    """
    Resample the dataframe so that there is an entry for each timestamp according to the aggregation
    rule: prioritization of main characters.

    file_path: the path of an emotion annotation file (csv)
    unit: resolution, default to 2 sec
    end_time: a timestamp larger than the length of the movie would work

    Returns: None, saves the tsv in a new file.
    """
    project_root = cfg.project_root

    # Path to the CSV file
    path = os.path.join(project_root, rel_file_path)
    df = pd.read_csv(path)

    # Main characters for prioritization
    main_characters = ["FORREST", "FORRESTVO", "JENNY", "DAN", "BUBBA", "MRSGUMP"]

    # Resample the dataframe, breaking rows into 2-sec intervals
    rows = []
    for _, row in df.iterrows():
        timestamps = range(int(row['start']), int(row['end']), unit)
        for timestamp in timestamps:
            new_row = row.copy()
            new_row['offset'] = timestamp  # Rename 'start' to 'offset' and update the timestamp
            rows.append(new_row)

    resampled_df = pd.DataFrame(rows)

    # Create a complete set of timestamps from 0 to end_time at the unit interval
    complete_timestamps = pd.DataFrame({'offset': range(0, end_time + unit, unit)})  # Renaming 'start' to 'offset'
    
    # Merge resampled data with complete timestamps
    df = pd.merge(complete_timestamps, resampled_df, on='offset', how='left').fillna({
        'emotion': 'NONE',  # Fill missing emotions with 'NONE'
        'character': '',  # Keep character empty if no data
    })

    # Prioritize main characters
    df['rank'] = pd.Categorical(df['character'], categories=main_characters, ordered=True)

    # Sort by offset time and main character rank
    df_sorted = df.sort_values(by=['offset', 'rank'])

    # Keep only the highest priority character per timestamp
    df_unique = df_sorted.drop_duplicates(subset=['offset'], keep='first')

    # Drop unnecessary columns and save the result
    df_unique = df_unique.drop(columns=['rank'])
    output_path = os.path.join(project_root, "data", "resampled_annotations", f"av1o{observer_index}_resampled.tsv")
    df_unique.to_csv(output_path, sep='\t', index=False)

    print(f"Resampled data saved to {output_path}")


if __name__ == "__main__":
    resample_events_prioritize()
