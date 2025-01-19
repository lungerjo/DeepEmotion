import zarr
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path
import hydra
import nibabel as nib
import pandas as pd
import numpy as np

# Directly embedding a minimal version of CrossSubjectDataset here
class CrossSubjectDataset:
    def __init__(self, cfg: DictConfig):
        self.data_path = Path(cfg.data.data_path).resolve()
        self.label_path = Path(cfg.data.label_path).resolve()
        self.subjects = cfg.data.subjects
        self.file_pattern_template = cfg.data.file_pattern_template
        self.sessions = cfg.data.sessions
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for alignment
        self.verbose = cfg.verbose
        self.emotion_idx = cfg.data.emotion_idx
        self.normalization = cfg.data.normalization
        self.observer_labels = pd.read_csv(self.label_path, sep='\t')

        # Find data files and get number of timepoints
        self.volume_paths_per_session, self.num_timepoints_per_volume = self._load_data_info()

        # Flatten the data files and num_timepoints lists
        self.data_files = []
        self.num_timepoints = []
        for volume_paths, num_timepoints_per_volume in zip(self.volume_paths_per_session, self.num_timepoints_per_volume):
            self.data_files.extend(volume_paths)
            self.num_timepoints.extend(num_timepoints_per_volume)

        if self.verbose:
            print(f"len(self.data_files): {len(self.data_files)}")
            print(f"ex {self.data_files[:5]}")
            print(f"len(self.num_timepoints): {len(self.num_timepoints)}")
            print(f"ex {self.num_timepoints[:5]}")

        # Align labels and filter out 'NONE' labels
        self.aligned_labels = self._align_labels()
        if self.verbose:
            print(f"aligned_labels: shape {self.aligned_labels.shape[0]}")
            print(self.aligned_labels.head())
        self.aligned_labels = self.aligned_labels[self.aligned_labels['emotion'] != 'NONE'].reset_index(drop=True)
        if self.verbose:
            print(f"filtered_labels: shape {self.aligned_labels.shape[0]}")
            print(self.aligned_labels.head())

        # Create index mappings between labels and data slices
        self.index_mappings = self._create_index_mappings()

    def _align_labels(self):
        """
        Align the observer's labels based on the provided session_offsets and the first subject's timepoints.
        """
        aligned_labels = []
        TR = 2  # seconds per TR

        for session_idx, volume_num_timepoints in enumerate(self.num_timepoints_per_volume):
            session_timepoints = volume_num_timepoints[0]
            session_start = self.session_offsets[session_idx]

            if session_idx < len(self.session_offsets) - 1:
                session_end = self.session_offsets[session_idx + 1]
            else:
                session_end = session_start + session_timepoints * TR

            if self.verbose:
                print(f"session_{session_idx}_start {session_start}, end {session_end}")

            # Filter labels within this session's time range
            session_labels = self.observer_labels[
                (self.observer_labels['offset'] >= session_start) &
                (self.observer_labels['offset'] < session_end)
            ].copy()

            aligned_labels.append(session_labels)

        aligned_labels = pd.concat(aligned_labels, ignore_index=True)

        return aligned_labels

    def _find_files(self):
        """Searches for files in all subject directories and runs based on the file pattern."""
        session_files = []
        if self.verbose:
            print(f"Finding files")
        for session in self.sessions:
            subject_files = []
            for subject in self.subjects:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                if self.verbose:
                    print(f"Looking in {subject_dir}")
                file_pattern = self.file_pattern_template.format(session)
                matched_files = list(subject_dir.rglob(file_pattern))
                subject_files.extend(matched_files)
            session_files.append(subject_files)
        if self.verbose:
            print(f"{sum(len(subject_files) for subject_files in session_files)} volumes found")
        return session_files

    def _load_data_info(self):
        """Gets the number of timepoints for each data file without loading data into memory."""
        session_files = self._find_files()
        volume_paths_per_session = []
        num_timepoints_per_volume = []

        for session_idx, subject_files in enumerate(session_files):
            volume_paths = []
            volume_num_timepoints = []
            for volume_path in subject_files:
                nii_img = nib.load(str(volume_path))
                shape = nii_img.shape  # (x, y, z, t)
                volume_num_timepoints.append(shape[-1])  # Number of timepoints (t)
                volume_paths.append(volume_path)
            volume_paths_per_session.append(volume_paths)
            num_timepoints_per_volume.append(volume_num_timepoints)

        if self.verbose:
            print(f"num_timepoints_per_volume {num_timepoints_per_volume}")

        return volume_paths_per_session, num_timepoints_per_volume

    def _create_index_mappings(self):
        """
        For each session, map label offsets to a local time index in [0, session_timepoints).
        Then apply that same row index to ALL volumes in that session (since they share the same time range).
        We only increment the global volume index after we've handled all volumes in the session.
        """
        index_mappings = []
        TR = 2  # seconds per TR

        # This will track how many volumes we've already placed in the 'flattened' list
        # after processing previous sessions.
        global_volume_offset = 0

        for session_idx, (volume_paths, volume_lengths) in enumerate(
            zip(self.volume_paths_per_session, self.num_timepoints_per_volume)
        ):
            # How many timepoints does each volume in this session have?
            # (They should all be the same, e.g. 451, 438, etc.)
            session_timepoints = volume_lengths[0]
            
            # Convert session_offsets to a time range for this session
            session_start = self.session_offsets[session_idx]
            if session_idx < len(self.session_offsets) - 1:
                next_session_start = self.session_offsets[session_idx + 1]
            else:
                # last session: just add TR * timepoints
                next_session_start = session_start + session_timepoints * TR

            # Filter labels that belong to this session’s time window
            session_mask = (
                (self.aligned_labels['offset'] >= session_start)
                & (self.aligned_labels['offset'] < next_session_start)
            )
            session_labels = self.aligned_labels[session_mask]

            # For each label event in this session:
            for row_idx, label_row in session_labels.iterrows():
                offset = label_row['offset']
                label_str = label_row['emotion']
                label_idx = self.emotion_idx[label_str]

                # Convert offset to a local time index in [0, session_timepoints)
                local_time_idx = int((offset - session_start) / TR)
                # Skip if it falls outside the actual volume length
                if not (0 <= local_time_idx < session_timepoints):
                    continue

                # Assign that same time index to EVERY volume in this session
                for vol_idx, n_tp in enumerate(volume_lengths):
                    # Just in case volumes differ slightly in actual length:
                    if local_time_idx >= n_tp:
                        continue

                    # data_file_idx is how we map back into the *flattened* list
                    data_file_idx = global_volume_offset + vol_idx

                    index_mappings.append({
                        'aligned_label_idx': row_idx,
                        'data_file_idx': data_file_idx,
                        'row_idx': local_time_idx,
                        'label_idx': label_idx
                    })

            # After finishing session_idx, move our global_volume_offset
            global_volume_offset += len(volume_paths)

        if self.verbose:
            print(f"mappings created: {len(index_mappings)}")
            for mapping in index_mappings[:5]:
                print(mapping)

        return index_mappings



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

    # Add session, session_idx, and global_idx columns
    merged_labels['session'] = merged_labels['file_index'].apply(lambda fi: file_to_session[fi])
    merged_labels['session_idx'] = merged_labels['session'].apply(lambda s: session_map[s])
    merged_labels['global_idx'] = np.arange(len(merged_labels))

    if cfg.verbose:
        print("Merged labels with metadata:")
        print(merged_labels.head())

    # Convert merged_labels to CSV and store
    csv_str = merged_labels.to_csv(index=False, sep='\t')

    if cfg.verbose:
        print("Data and label arrays prepared.")
        print(f"Spatial shape: ({x}, {y}, {z})")
        print(f"Time max (t_max): {t_max}")
        print(f"Creating Zarr store at {output_zarr_path}...")

    # Create Zarr store
    store = zarr.group(output_zarr_path, overwrite=True)
    # Store the enriched metadata
    store.attrs['aligned_labels'] = csv_str

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

    # Write all volumes
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

    if cfg.verbose:
        print("Storing metadata: file paths, subjects, sessions...")

    file_paths = np.array([str(p) for p in data_files], dtype=object)

    # Determine max lengths for strings
    max_path_len = max(len(p) for p in file_paths)
    max_sub_len = max(len(s) for s in subjects)
    max_ses_len = max(len(s) for s in sessions)
    max_subj_len = max(len(s) for s in file_to_subject if s is not None)
    max_file_ses_len = max(len(s) for s in file_to_session if s is not None)

    # Store arrays with fixed-length Unicode dtype
    store.create_dataset("file_paths", data=file_paths.astype(f'U{max_path_len}'), shape=(n_files,))
    store.create_dataset("subject_ids", data=np.array(subjects, dtype=f'U{max_sub_len}'), shape=(len(subjects),))
    store.create_dataset("session_ids", data=np.array(sessions, dtype=f'U{max_ses_len}'), shape=(len(sessions),))
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
        t_idx = mapping['row_idx']
        l_idx = mapping['label_idx']
        label_array[f_idx, t_idx] = l_idx

    # Store valid_timepoints
    store.create_dataset("valid_timepoints", data=valid_timepoints, shape=(n_files,), dtype='int32')

    # Store emotion mapping as attributes
    emotions = list(emotion_idx.keys())
    store.attrs['emotions'] = emotions

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
