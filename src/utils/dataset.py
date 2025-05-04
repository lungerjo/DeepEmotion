import zarr
import torch
import pandas as pd
from io import StringIO
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import Dataset
import nibabel as nib
from collections import defaultdict

class StudyForrestVolumeIndexer:
    def __init__(self, cfg: DictConfig):

        self.data_path = (Path(cfg.project_root) / cfg.data.data_path).resolve()
        self.subjects = cfg.data.subjects
        self.sessions = cfg.data.sessions
        self.file_pattern_template = cfg.data.file_pattern_template
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for alignment
        self.verbose = cfg.verbose
        self.classification_emotion_idx = cfg.data.classification_emotion_idx

        # Find data files and get number of timepoints
        self.volume_paths, self.volume_to_session = self._get_volume_paths()
        self.volume_lengths = self._get_volume_lengths(self.volume_paths)

        self.classification_labels = pd.read_csv((Path(cfg.project_root) / cfg.data.classification_label_path).resolve(), sep='\t')
        self.soft_classification_labels = pd.read_csv((Path(cfg.project_root) / cfg.data.soft_classification_label_path).resolve(), sep='\t')
        if cfg.data.label_mode == "classification":
            self.selected_labels = self.classification_labels
        elif cfg.data.label_mode == "soft_classification":
            self.selected_labels = self.soft_classification_labels
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        # Align labels
        self.aligned_labels_df = self._get_aligned_labels_df()

        # Create index mappings between labels and data slices
        self.index_mappings = self._create_index_mappings()

    def _get_aligned_labels_df(self):
        """
        Align the observer's labels based on session_offsets and volume-to-session mapping.
        Returns:
            pd.DataFrame: Aligned labels within the fMRI acquisition range.
        """
        aligned_labels = []
        TR = 2  # seconds per TR

        # Group volume indices by session
        session_to_volumes = defaultdict(list)
        for i, sess_idx in enumerate(self.volume_to_session):
            session_to_volumes[sess_idx].append(i)

        for session_idx in range(len(self.session_offsets)):
            volume_indices = session_to_volumes[session_idx]
            if not volume_indices:
                continue

            session_start = self.session_offsets[session_idx]
            if session_idx < len(self.session_offsets) - 1:
                session_end = self.session_offsets[session_idx + 1]
            else:
                # fallback: estimate using the timepoints of the first volume in the session
                first_volume_idx = volume_indices[0]
                session_timepoints = self.volume_lengths[first_volume_idx]
                session_end = session_start + session_timepoints * TR

            if self.verbose:
                print(f"session_{session_idx}_start {session_start}, end {session_end}")

            session_labels = self.selected_labels[
                (self.selected_labels['offset'] >= session_start) &
                (self.selected_labels['offset'] < session_end)
            ].copy()

            aligned_labels.append(session_labels)

        return pd.concat(aligned_labels, ignore_index=True)


    def _get_volume_lengths(self, paths: list[Path]) -> list[int]:
        return [nib.load(str(p)).shape[-1] for p in paths]

    def _get_volume_paths(self):
        paths = []
        session_indices = []

        for session_idx, session in enumerate(self.sessions):
            for subject in self.subjects:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                file_pattern = self.file_pattern_template.format(session)
                matched = list(subject_dir.rglob(file_pattern))
                paths.extend(matched)
                session_indices.extend([session_idx] * len(matched))

        return paths, session_indices

    def _create_index_mappings(self):
        """
        For each session, map label offsets to a local time index in [0, session_timepoints).
        Then apply that same row index to ALL volumes in that session (since they share the same time range).
        """
        index_mappings = []
        TR = 2  # seconds per TR

        # Group volume indices by session
        session_to_volumes = defaultdict(list)
        for i, sess_idx in enumerate(self.volume_to_session):
            session_to_volumes[sess_idx].append(i)

        for session_idx in range(len(self.session_offsets)):
            volume_indices = session_to_volumes[session_idx]
            if not volume_indices:
                continue

            session_start = self.session_offsets[session_idx]
            if session_idx < len(self.session_offsets) - 1:
                session_end = self.session_offsets[session_idx + 1]
            else:
                session_timepoints = self.volume_lengths[volume_indices[0]]
                session_end = session_start + session_timepoints * TR

            # Labels that fall within this session's time window
            session_mask = (
                (self.aligned_labels_df['offset'] >= session_start) &
                (self.aligned_labels_df['offset'] < session_end)
            )
            session_labels = self.aligned_labels_df[session_mask]

            for row_idx, label_row in session_labels.iterrows():
                offset = label_row['offset']
                if self.label_mode == "classification":
                    label_str = label_row['emotion']
                    label_idx = self.classification_emotion_idx[label_str]
                elif self.label_mode == "soft_classification":
                    label_idx = -1
                else:
                    raise ValueError(f"Unsupported label mode: {self.label_mode}")

                local_time_idx = int((offset - session_start) / TR)

                for vol_idx in volume_indices:
                    n_tp = self.volume_lengths[vol_idx]
                    if 0 <= local_time_idx < n_tp:
                        index_mappings.append({
                            'aligned_label_idx': row_idx,
                            'data_file_idx': vol_idx,
                            'row_idx': local_time_idx,
                            'label_idx': label_idx
                        })

        if self.verbose:
            print(f"mappings created: {len(index_mappings)}")
            for mapping in index_mappings[:5]:
                print(mapping)

        return index_mappings


    
class ZarrDataset(Dataset):
    def __init__(self, zarr_path: str, label_mode: str, debug: bool, cfg: DictConfig = None):
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
        self.file_to_subject = self.store["file_to_subject"][:]
        self.file_to_session = self.store["file_to_session"][:]
        self.subject_ids = self.store["subject_ids"][:]
        self.session_ids = self.store["session_ids"][:]
        self.allowed_subjects = set(cfg.data.subjects)
        self.allowed_sessions = set(cfg.data.sessions)

        if self.debug:
            print(f"[DEBUG] allowed_subjects {self.allowed_subjects}")
        if self.debug:
            print(f"[DEBUG] allowed_sessions {self.allowed_sessions}")

        keep_files = []
        for i in range(len(self.file_to_subject)):
            subj = self.file_to_subject[i]
            sess = self.file_to_session[i]
            if subj in self.allowed_subjects and sess in self.allowed_sessions:
                keep_files.append(i)
        keep_files = set(keep_files)

        mask = np.array([f in keep_files for f, _ in self.valid_indices])
        self.valid_indices = self.valid_indices[mask]


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
