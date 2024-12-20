import zarr
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path
import hydra
import nibabel as nib

from utils.dataset import CrossSubjectDataset  # Adjust this import as necessary

def write_zarr_dataset(cfg: DictConfig, output_zarr_path: str):
    if cfg.verbose:
        print("Initializing CrossSubjectDataset for alignment and indexing...")

    # Instantiate the dataset to get alignment, indexing, and metadata.
    dataset = CrossSubjectDataset(cfg)
    if cfg.verbose:
        print("Dataset indexing and alignment complete.")
        print(f"Number of files: {len(dataset.data_files)}")
        print(f"Number of valid samples (excluding 'NONE' labels): {len(dataset)}")
        print(f"Subjects: {dataset.subjects}")
        print(f"Sessions: {dataset.sessions}")
        print("Available emotions and their indices:")
        for emotion, idx in dataset.emotion_idx.items():
            print(f"  {emotion}: {idx}")

    # Extract metadata from the dataset
    data_files = dataset.data_files       # List[Path]
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

    if cfg.verbose:
        print("Data and label arrays prepared.")
        print(f"Spatial shape: ({x}, {y}, {z})")
        print(f"Time max (t_max): {t_max}")
        print(f"Creating Zarr store at {output_zarr_path}...")

    # Create Zarr store
    store = zarr.group(output_zarr_path, overwrite=True)

    # Choose chunking strategy
    chunk_size = (1, x, y, z, 50)
    if cfg.verbose:
        print(f"Chunk size chosen: {chunk_size}")

    # Create the main data dataset
    data_zarr = store.create_dataset(
        "data",
        shape=(n_files, x, y, z, t_max),
        chunks=chunk_size,
        dtype='float32',
        compressor=None
    )

    if cfg.verbose:
        print("Writing data to Zarr dataset one file at a time...")

    # Keep track of valid timepoints per file
    valid_timepoints = np.zeros(n_files, dtype='int32')

    for i, file_path in enumerate(data_files):
        if cfg.verbose:
            print(f"  Loading file {i}: {file_path}")

        # Load the entire file into memory as a float32 volume
        nii_img = nib.load(str(file_path))
        volume = nii_img.get_fdata(dtype=np.float32)  # shape: (x, y, z, t_current)
        t_current = volume.shape[-1]

        # Record the valid timepoints for this file
        valid_timepoints[i] = t_current

        # Optional normalization
        if cfg.data.normalization:
            mean_val = volume.mean()
            std_val = volume.std() + 1e-5
            volume = (volume - mean_val) / std_val

        if cfg.verbose:
            print(f"  Writing file {i} with shape {volume.shape} into data_zarr, padded up to {t_max} timepoints.")

        # Write to Zarr
        data_zarr[i, ..., :t_current] = volume
        # Rest remains zero-padded

        # Free memory
        del volume
        del nii_img

    # Store subject/session metadata
    subject_ids = np.array(subjects, dtype=object)
    session_ids = np.array(sessions, dtype=object)

    if cfg.verbose:
        print("Storing metadata: file paths, subjects, sessions...")

    file_paths = np.array([str(p) for p in data_files], dtype=object)

    # Map each file to a subject and session
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

    # Determine max lengths for strings before converting to fixed-length Unicode
    max_path_len = max(len(p) for p in file_paths)
    max_sub_len = max(len(s) for s in subject_ids)
    max_ses_len = max(len(s) for s in session_ids)
    max_subj_len = max(len(s) for s in file_to_subject if s is not None)
    max_file_ses_len = max(len(s) for s in file_to_session if s is not None)

    # Store arrays with fixed-length Unicode dtype
    store.create_dataset("file_paths", data=file_paths.astype(f'U{max_path_len}'), shape=(n_files,))
    store.create_dataset("subject_ids", data=subject_ids.astype(f'U{max_sub_len}'), shape=(len(subjects),))
    store.create_dataset("session_ids", data=session_ids.astype(f'U{max_ses_len}'), shape=(len(sessions),))
    store.create_dataset("file_to_subject", data=file_to_subject.astype(f'U{max_subj_len}'), shape=(n_files,))
    store.create_dataset("file_to_session", data=file_to_session.astype(f'U{max_file_ses_len}'), shape=(n_files,))

    # Create the label dataset
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
        t_idx = mapping['time_idx']
        l_idx = mapping['label_idx']
        label_array[f_idx, t_idx] = l_idx

    # Store valid_timepoints
    store.create_dataset("valid_timepoints", data=valid_timepoints, shape=(n_files,), dtype='int32')

    # Store emotion mapping as attributes
    emotions = list(emotion_idx.keys())
    store.attrs['emotions'] = emotions

    if cfg.verbose:
        print("Storing aligned labels as attribute (CSV format)...")
    csv_str = aligned_labels.to_csv(index=False, sep='\t')
    store.attrs['aligned_labels'] = csv_str

    if cfg.verbose:
        print(f"Zarr dataset successfully written to {output_zarr_path}!")


@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    output_path = str((Path(cfg.project_root) / "dataset.zarr").resolve())
    if cfg.verbose:
        print("Starting Zarr dataset creation...")
    write_zarr_dataset(cfg, output_path)
    if cfg.verbose:
        print("Zarr dataset creation completed.")

if __name__ == "__main__":
    main()
