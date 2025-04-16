import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

# Modify this to the full absolute path to your input CSV file
absolute_file_path = "/home/paperspace/DeepEmotion/data/annotations/src/emotions/data/raw/ao1o03.csv"
unit = 2
end_time = 7031

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def resample_events_prioritize(cfg: DictConfig):
    """
    Resample the dataframe so that there is an entry for each timestamp according to the aggregation
    rule: prioritization of main characters.

    absolute_file_path: full path of an emotion annotation file (csv)
    unit: resolution, default to 2 sec
    end_time: a timestamp larger than the length of the movie would work

    Returns: None, saves the tsv in a new file.
    """
    df = pd.read_csv(absolute_file_path)

    # Main characters for prioritization
    main_characters = ["FORREST", "FORRESTVO", "JENNY", "DAN", "BUBBA", "MRSGUMP"]

    # Resample the dataframe, breaking rows into 2-sec intervals
    rows = []
    for _, row in df.iterrows():
        timestamps = range(int(row['start']), int(row['end']), unit)
        for timestamp in timestamps:
            new_row = row.copy()
            new_row['offset'] = timestamp
            rows.append(new_row)

    resampled_df = pd.DataFrame(rows)

    # Create a complete set of timestamps
    complete_timestamps = pd.DataFrame({'offset': range(0, end_time + unit, unit)})

    # Merge resampled data with complete timestamps
    df = pd.merge(complete_timestamps, resampled_df, on='offset', how='left').fillna({
        'emotion': 'NONE',
        'character': '',
    })

    df['rank'] = pd.Categorical(df['character'], categories=main_characters, ordered=True)

    # Sort by time and rank, keep top priority
    df_sorted = df.sort_values(by=['offset', 'rank'])
    df_unique = df_sorted.drop_duplicates(subset=['offset'], keep='first')
    df_unique = df_unique.drop(columns=['rank'])

    # Construct an output path next to the input file
    output_path = os.path.join(
        os.path.dirname(absolute_file_path).replace("raw", "resampled"),
        os.path.basename(absolute_file_path).replace(".csv", "_resampled.tsv")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_unique.to_csv(output_path, sep='\t', index=False)

    print(f"Resampled data saved to {output_path}")


if __name__ == "__main__":
    resample_events_prioritize()
