import zarr
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path
import hydra
import nibabel as nib
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.dataset import CrossSubjectDataset

def write_zarr_dataset(cfg: DictConfig, output_zarr_path: str):
    if cfg.verbose:
        print("Initializing CrossSubjectDataset for alignment and indexing...")

    # Instantiate the dataset to get alignment, indexing, and metadata.
    dataset = CrossSubjectDataset(cfg)
    if cfg.verbose:
        print("Dataset indexing and alignment complete.")
        print(f"Number of files: {len(dataset.data_files)}")
        print(f"Number of valid samples (excluding 'NONE' labels): {len(dataset.index_mappings)}")
        print(f"Subjects: {dataset.subjects}")
        print(f"Sessions: {dataset.sessions}")

    # Extract metadata from the dataset
    data_files = dataset.data_files
    num_timepoints = dataset.num_timepoints
    index_mappings = dataset.index_mappings
    aligned_labels = dataset.aligned_labels
    subjects = dataset.subjects
    sessions = dataset.sessions
    emotion_idx = dataset.emotion_idx

    n_files = len(data_files)
    if n_files == 0:
        raise ValueError("No data files found.")

    # Determine spatial shape from the first file
    if cfg.verbose:
        print("Determining spatial shape from the first file...")
    first_nii = nib.load(str(data_files[0]))
    x, y, z, _ = first_nii.shape
    del first_nii

    t_max = max(num_timepoints)

    # Before merging labels, define file_to_subject and file_to_session arrays
    file_to_subject = []
    file_to_session = []
    for fpath in data_files:
        sub = None
        ses = None
        for s in subjects:
            if s in fpath.parts:
                sub = s
                break
        for ssn in sessions:
            if ssn in fpath.name:
                ses = ssn
                break
        file_to_subject.append(sub)
        file_to_session.append(ses)

    file_to_subject = np.array(file_to_subject, dtype=object)
    file_to_session = np.array(file_to_session, dtype=object)

    # Merge index_mappings with aligned_labels
    index_mappings_df = pd.DataFrame(index_mappings)
    aligned_labels_reset = aligned_labels.reset_index(drop=False).rename(columns={'index': 'aligned_label_idx'})
    merged_labels = pd.merge(index_mappings_df, aligned_labels_reset, on='aligned_label_idx', how='left')

    # Rename columns for clarity
    merged_labels = merged_labels.rename(columns={
        'data_file_idx': 'file_index',
        'row_idx': 'row_index',
        'offset': 'time_offset'
    })

    # Create a session index mapping
    session_map = {s: i for i, s in enumerate(sessions)}
    merged_labels['session'] = merged_labels['file_index'].apply(lambda fi: file_to_session[fi])
    merged_labels['session_idx'] = merged_labels['session'].apply(lambda s: session_map[s])
    merged_labels['global_idx'] = np.arange(len(merged_labels))

    # Add regression metadata to aligned_labels CSV
    if "regression_label_path" in cfg.data:
        reg_df = pd.read_csv(Path(cfg.data.regression_label_path), sep="\t")
        merged_labels = pd.merge(merged_labels, reg_df, left_on="time_offset", right_on="offset", how="left")

    if cfg.verbose:
        print("Merged labels with metadata:")
        print(merged_labels.head())

    csv_str = merged_labels.to_csv(index=False, sep='\t')

    if cfg.verbose:
        print("Data and label arrays prepared.")
        print(f"Spatial shape: ({x}, {y}, {z})")
        print(f"Time max (t_max): {t_max}")
        print(f"Creating Zarr store at {output_zarr_path}...")

    store = zarr.group(output_zarr_path, overwrite=True)
    store.attrs['aligned_labels'] = csv_str

    chunk_size = (1, x, y, z, 1)
    if cfg.verbose:
        print(f"Chunk size chosen: {chunk_size}")

    data_zarr = store.create_dataset(
        "data",
        shape=(n_files, x, y, z, t_max),
        chunks=chunk_size,
        dtype='float32',
        compressor=None
    )

    if cfg.verbose:
        print("Writing data to Zarr dataset one file at a time...")

    valid_timepoints = np.zeros(n_files, dtype='int32')

    for i, file_path in enumerate(data_files):
        if cfg.verbose:
            print(f"  Loading file {i}: {file_path}")

        nii_img = nib.load(str(file_path))
        volume = nii_img.get_fdata(dtype=np.float32)
        t_current = volume.shape[-1]
        valid_timepoints[i] = t_current

        if cfg.data.normalization:
            mean_val = volume.mean()
            std_val = volume.std() + 1e-5
            volume = (volume - mean_val) / std_val

        data_zarr[i, ..., :t_current] = volume
        del volume
        del nii_img

    if cfg.verbose:
        print("Storing metadata: file paths, subjects, sessions...")

    file_paths = np.array([str(p) for p in data_files], dtype=object)

    max_path_len = max(len(p) for p in file_paths)
    max_sub_len = max(len(s) for s in subjects)
    max_ses_len = max(len(s) for s in sessions)
    max_subj_len = max(len(s) for s in file_to_subject if s is not None)
    max_file_ses_len = max(len(s) for s in file_to_session if s is not None)

    store.create_dataset("file_paths", data=file_paths.astype(f'U{max_path_len}'), shape=(n_files,))
    store.create_dataset("subject_ids", data=np.array(subjects, dtype=f'U{max_sub_len}'), shape=(len(subjects),))
    store.create_dataset("session_ids", data=np.array(sessions, dtype=f'U{max_ses_len}'), shape=(len(sessions),))
    store.create_dataset("file_to_subject", data=file_to_subject.astype(f'U{max_subj_len}'), shape=(n_files,))
    store.create_dataset("file_to_session", data=file_to_session.astype(f'U{max_file_ses_len}'), shape=(n_files,))

    if cfg.verbose:
        print("Creating label dataset...")

    label_array = store.create_dataset(
        "labels",
        shape=(n_files, t_max),
        chunks=(1, 50),
        dtype='int32',
        fill_value=-1
    )

    if cfg.verbose:
        print("Filling label array with aligned labels...")
    for mapping in index_mappings:
        f_idx = mapping['data_file_idx']
        t_idx = mapping['row_idx']
        l_idx = mapping['label_idx']
        label_array[f_idx, t_idx] = l_idx

    store.create_dataset("valid_timepoints", data=valid_timepoints, shape=(n_files,), dtype='int32')
    emotions = list(emotion_idx.keys())
    store.attrs['emotions'] = emotions

    if cfg.data.get("regression_label_path"):
        if cfg.verbose:
            print("Adding regression_labels to Zarr...")

        reg_df = pd.read_csv(Path(cfg.data.regression_label_path).expanduser(), sep="\t")
        regression_columns = [col for col in reg_df.columns if col != 'offset']

        # Merge regression values into the aligned_labels metadata
        regression_merged = pd.merge(merged_labels, reg_df, left_on='time_offset', right_on='offset', how='left')

        # Save to aligned_labels attr
        csv_str = regression_merged.to_csv(index=False, sep='\t')
        store.attrs['aligned_labels'] = csv_str

        # Write regression tensor
        num_regression_dims = len(regression_columns)
        regression_labels = store.create_dataset(
            "regression_labels",
            shape=(n_files, t_max, num_regression_dims),
            chunks=(1, 50, num_regression_dims),
            dtype='float32',
            fill_value=np.nan
        )

        for _, row in regression_merged.iterrows():
            if pd.isna(row["file_index"]) or pd.isna(row["row_index"]):
                continue  # skip rows we don't have index info for
            f_idx = int(row["file_index"])
            t_idx = int(row["row_index"])
            values = row[regression_columns].values.astype(np.float32)
            regression_labels[f_idx, t_idx, :] = values

    if cfg.verbose:
        print(f"Zarr dataset successfully written to {output_zarr_path}!")


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:

    if cfg.verbose:
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        print(f"[DEBUG] Hydra project root: {cfg.project_root}")
        print(f"[DEBUG] Config-resolved data path: {cfg.data.data_path}")
        print(f"[DEBUG] Full resolved data path: {(Path(cfg.project_root) / cfg.data.data_path).resolve()}")
    output_path = str((Path(cfg.data.zarr_dir_path) / "pool_emotions").resolve())
    if cfg.verbose:
        print("Starting Zarr dataset creation...")
    write_zarr_dataset(cfg, output_path)
    if cfg.verbose:
        print("Zarr dataset creation completed.")

if __name__ == "__main__":
    main()
