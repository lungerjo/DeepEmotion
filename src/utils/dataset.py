import zarr
import torch
import pandas as pd
from io import StringIO
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import Dataset
import nibabel as nib

# Directly embedding a minimal version of CrossSubjectDataset here
class CrossSubjectDataset:
    def __init__(self, cfg: DictConfig):
        self.data_path = (Path(cfg.project_root) / cfg.data.data_path).resolve()
        self.label_mode = cfg.data.label_mode
        self.classification_labels = pd.read_csv((Path(cfg.project_root) / cfg.data.classification_label_path).resolve(), sep='\t')
        self.soft_classification_labels = pd.read_csv((Path(cfg.project_root) / cfg.data.soft_classification_label_path).resolve(), sep='\t')

        if self.label_mode == "classification":
            self.observer_labels = self.classification_labels
        elif self.label_mode == "soft_classification":
            self.observer_labels = self.soft_classification_labels
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        self.subjects = cfg.data.subjects
        self.file_pattern_template = cfg.data.file_pattern_template
        self.sessions = cfg.data.sessions
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for alignment
        self.verbose = cfg.verbose
        self.classification_emotion_idx = cfg.data.classification_emotion_idx
        self.normalization = cfg.data.normalization

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
        if self.label_mode == "classification":
            self.aligned_labels = self.aligned_labels[self.aligned_labels['emotion'] != 'NONE'].reset_index(drop=True)

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
        if self.verbose:
            print(f"[DEBUG] Finding files using pattern: {self.file_pattern_template}")
        session_files = []

        for session in self.sessions:
            if self.verbose:
                print(f"[DEBUG] Looking for session: {session}")
            subject_files = []
            for subject in self.subjects:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                file_pattern = self.file_pattern_template.format(session)

                if self.verbose:
                    print(f"[DEBUG] Looking in directory: {subject_dir}")
                    print(f"[DEBUG] Using file pattern: {file_pattern}")

                matched_files = list(subject_dir.rglob(file_pattern))
                if self.verbose:
                    print(f"[DEBUG] Found {len(matched_files)} files for subject {subject} session {session}")

                subject_files.extend(matched_files)
            session_files.append(subject_files)

        return session_files

    def _load_data_info(self):
        """Gets the number of timepoints for each data file without loading data into memory."""
        session_files = self._find_files()
        volume_paths_per_session = []
        num_timepoints_per_volume = []

        if self.verbose:
            print(f"[DEBUG] Loading data info for {len(session_files)} sessions")

        for session_idx, subject_files in enumerate(session_files):
            if self.verbose:
                print(f"[DEBUG] Session {session_idx}: Found {len(subject_files)} files")
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
                if self.label_mode == "classification":
                    label_str = label_row['emotion']
                    label_idx = self.classification_emotion_idx[label_str]
                elif self.label_mode == "soft_classification":
                    label_idx = -1  # Not used for soft_classificationn, but needs to be present
                else:
                    raise ValueError(f"Unsupported label mode: {self.label_mode}")


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
    
class ZarrDataset(Dataset):
    def __init__(self, zarr_path: str, label_mode: str, debug: bool):
        self.debug = debug
        self.store = zarr.open(zarr_path, mode='r')
        if self.debug:
            print(f"[DEBUG] list(self.store.keys()): {list(self.store.keys())}")

        self.label_mode = label_mode
        self.data = self.store['data']
        self.valid_timepoints = self.store['valid_timepoints'][:]
        self.file_paths = self.store['file_paths'][:]
        self.subject_ids = self.store['subject_ids'][:]
        self.session_ids = self.store['session_ids'][:]
        self.file_to_subject = self.store['file_to_subject'][:]
        self.file_to_session = self.store['file_to_session'][:]
        self.emotions = self.store.attrs.get('emotions', [])
        self.aligned_labels_csv = self.store.attrs.get('aligned_labels', None)

        if label_mode == "classification":
            if "classification_labels" not in self.store:
                raise ValueError("Zarr store missing classification labels.")
            self.labels = self.store['classification_labels']
        elif label_mode == "soft_classification":
            if "soft_classification_labels" not in self.store:
                raise ValueError("Zarr store missing soft_classification labels.")
            self.labels = self.store['soft_classification_labels']
        else:
            raise ValueError(f"Unsupported label mode: {label_mode}")
        if self.debug:
            print(f"[DEBUG] self.labels.shape: {self.labels.shape}")

        self.valid_indices = self.store["valid_indices"][:]

        if self.aligned_labels_csv:
            self.aligned_labels = pd.read_csv(StringIO(self.aligned_labels_csv), sep='\t')
        else:
            self.aligned_labels = None


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        # Retrieve the (volume_idx, time_idx) for this valid sample
        volume_idx, row_idx = self.valid_indices[idx]
        volume_idx = int(volume_idx)
        row_idx = int(row_idx)
        
        
        # Extract the data slice
        data_slice = self.data[volume_idx, :, :, :, row_idx]
        data_tensor = torch.from_numpy(data_slice)

        # Extract the label
        if self.label_mode == "classification":
            label = int(self.labels[volume_idx, row_idx])
            label_tensor = torch.tensor(label, dtype=torch.long)
        elif self.label_mode == "soft_classification":
            label = self.labels[volume_idx, row_idx, :]
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported label mode: {self.label_mode}")

        # Validate subject/session mapping
        subject = self.file_to_subject[volume_idx]
        session = self.file_to_session[volume_idx]
        assert subject in self.subject_ids, f"Subject {subject} not in subject_ids."
        assert session in self.session_ids, f"Session {session} not in session_ids."

        # Optional aligned metadata – default to numeric sentinels, never None
        global_idx = idx

        if self.aligned_labels is not None:
            subset = self.aligned_labels[
                (self.aligned_labels["file_index"] == volume_idx)
                & (self.aligned_labels["row_index"] == row_idx)
            ]
            if subset.empty:
                print(f"[ERROR] No aligned label found for volume_idx={volume_idx}, row_idx={row_idx}")
                print(f"[ERROR] Total aligned_labels shape: {self.aligned_labels.shape}")
                print(f"[ERROR] Unique file_index: {np.unique(self.aligned_labels['file_index'])[:10]}")
                print(f"[ERROR] Unique row_index: {np.unique(self.aligned_labels['row_index'])[:10]}")
                print(f"[ERROR] Sample aligned_labels head:\n{self.aligned_labels.head()}")
                raise IndexError(f"Missing global_idx for volume_idx={volume_idx}, row_idx={row_idx}")
            global_idx = int(subset["global_idx"].iloc[0])


        sample = {
            "global_idx": global_idx,
            "volume_idx": volume_idx,
            "local_index": row_idx,
            "data_tensor": data_tensor,
            "label_tensor": label_tensor,
            "file_path": self.file_paths[volume_idx],
            "subject": subject,
            "session": session,
        }

        return sample

class ContrastivePairDataset(Dataset):
    """
    Wraps a base dataset to yield anchor, positive, and negative samples
    based on label matching. 
    Positive pairs have the same label. Negatives differ in qlabel.
    """
    def __init__(self, base_dataset: ZarrDataset, seed: int = 123):
        super().__init__()
        self.base = base_dataset
        self.rng = np.random.default_rng(seed=seed)

        # Build a dict: label -> all valid indices with that label
        self.label_to_indices = {}
        for idx in range(len(self.base)):
            label = self.base[idx]['label_tensor'].item()
            self.label_to_indices.setdefault(label, []).append(idx)

        # Preconvert all labels to a list so we can quickly sample negative labels
        self.all_labels = sorted(self.label_to_indices.keys())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, anchor_idx):
        anchor_dict = self.base[anchor_idx]
        anchor_label = anchor_dict['label_tensor'].item()

        if hasattr(self.base, 'verbose') and self.base.verbose:
            print(f"[Dataset] Anchor idx: {anchor_idx}, label: {anchor_label}")

        # Positive sample
        pos_indices = self.label_to_indices[anchor_label]
        positive_idx = anchor_idx
        if len(pos_indices) > 1:
            while positive_idx == anchor_idx:
                positive_idx = self.rng.choice(pos_indices)
        positive_dict = self.base[positive_idx]

        # Negative sample
        negative_label = anchor_label
        while negative_label == anchor_label and len(self.all_labels) > 1:
            negative_label = self.rng.choice(self.all_labels)
        neg_indices = self.label_to_indices[negative_label]
        negative_idx = self.rng.choice(neg_indices)
        negative_dict = self.base[negative_idx]

        return {
            "anchor": anchor_dict["data_tensor"],
            "positive": positive_dict["data_tensor"],
            "negative": negative_dict["data_tensor"],
            "anchor_label": anchor_dict["label_tensor"],
            "positive_label": positive_dict["label_tensor"],
            "negative_label": negative_dict["label_tensor"]
        }
