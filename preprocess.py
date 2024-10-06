import os
import numpy as np
import pandas as pd


def resample_events(file_path, unit=1, end_time=8000):
    """
    resample the dataframe so that there is an entry for each timestamp, note that there
    could be timestamps with 0, 1, or 2 emotions recorded, so there might be duplicates rows
    or rows with all 0s
    
    unit: resolution, default to 1 sec
    end_time: a timestamp larger than the length of the movie would work
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
    filled_df = pd.merge(complete_timestamps, resampled_df, on='onset', how='left').fillna(0)
    filled_df['character'] = filled_df['character'].replace(0, '') 
    filled_df.to_csv(os.path.join("data", "resampled_annotations", file_path), sep='\t', index=False)


if __name__ == "__main__":
    resample_events("emotions_av_1s_events_run-2_events.tsv")

