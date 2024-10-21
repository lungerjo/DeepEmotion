import os
import numpy as np
import pandas as pd


def resample_events_prioritize(file_path, unit=1, end_time=8000):
    """
    Resample the dataframe so that there is an entry for each timestamp according to the aggregation
    rule: prioritization.
        prioritizing assumes subjects resonates only/mostly with the main characters

    file_path: the path of a emotion annotion file (tsv)
    unit: resolution, default to 1 sec
    end_time: a timestamp larger than the length of the movie would work
    
    Returns: None, saves the tsv in a new file
    """
    df = pd.read_csv(os.path.join("data", "annotations", file_path), sep='\t')
    rows = []  
    for _, row in df.iterrows():
        timestamps = range(int(row['onset']), int(row['onset']) + int(np.ceil(row['duration'] / unit)) * unit, unit)
        for timestamp in timestamps:
            new_row = row.copy()
            new_row['onset'] = timestamp
            rows.append(new_row)

    resampled_df = pd.DataFrame(rows)
    complete_timestamps = pd.DataFrame({'onset': range(0, end_time + unit, unit)})
    df = pd.merge(complete_timestamps, resampled_df, on='onset', how='left').fillna(0)
    df['character'] = df['character'].replace(0, '')

    # prioritization
    main_characters = ["FORREST", "FORRESTVO", "JENNY", "DAN", "BUBBA", "MRSGUMP"]  # assuming FORRESTVO is gump jr
    df['rank'] = pd.Categorical(df['character'], categories=main_characters, ordered=True)
    df_main = df.dropna(subset=['rank'])
    df_other = df[df['rank'].isna()]
    df_other = df_other.groupby('onset').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
    df_combined = pd.concat([df_main, df_other])
    df_sorted = df_combined.sort_values(by=['onset', 'rank'])
    df_unique = df_sorted.drop_duplicates(subset=['onset'], keep='first')
    df_unique = df_unique.drop(columns=['rank'])

    df_unique.to_csv(os.path.join("data", "resampled_annotations", file_path), sep='\t', index=False)


def resample_events_add(file_path, unit=1, end_time=8000):
    """
    Resample the dataframe so that there is an entry for each timestamp according to the aggregation
    rule: addition.
        addition assumes different regions in the brain fire simultaneously with different emotions;

    file_path: the path of a emotion annotion file (tsv)
    unit: resolution, default to 1 sec
    end_time: a timestamp larger than the length of the movie would work
    
    Returns: None, saves the tsv in a new file
    """
    df = pd.read_csv(os.path.join("data", "annotations", file_path), sep='\t')
    rows = []  
    for _, row in df.iterrows():
        timestamps = range(int(row['onset']), int(row['onset']) + int(np.ceil(row['duration'] / unit)) * unit, unit)
        for timestamp in timestamps:
            new_row = row.copy()
            new_row['onset'] = timestamp
            rows.append(new_row)

    resampled_df = pd.DataFrame(rows)
    complete_timestamps = pd.DataFrame({'onset': range(0, end_time + unit, unit)})
    df = pd.merge(complete_timestamps, resampled_df, on='onset', how='left').fillna(0)
    df['character'] = df['character'].replace(0, '')

    # addition
    df_one_hot = pd.get_dummies(df['character'])
    df_encoded = pd.concat([df, df_one_hot], axis=1)
    df_encoded = df_encoded.drop(columns=['character', 'duration'])
    df_grouped = df_encoded.groupby('onset').sum().reset_index()

    df_grouped.to_csv(os.path.join("data", "resampled_annotations", file_path), sep='\t', index=False)


if __name__ == "__main__":
    resample_events_add("emotion_audio_only.tsv")

