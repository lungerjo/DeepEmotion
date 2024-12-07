import nibabel as nib
import h5py
import hydra
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import os

def create_hdf5(cfg: DictConfig):
    # Load observer labels (the full annotation TSV)
    observer_labels = pd.read_csv(cfg.data.label_path, sep='\t')

    subjects = cfg.data.subjects
    sessions = cfg.data.sessions
    data_path = Path(cfg.data.data_path).resolve()
    file_pattern_template = cfg.data.file_pattern_template
    session_offsets = cfg.data.session_offsets
    emotion_idx = cfg.data.emotion_idx
    normalization = cfg.data.normalization
    verbose = cfg.verbose

    # First pass: compute num_timepoints for each subject-session to replicate offset logic
    num_timepoints = []
    all_session_paths = []
    for subject in subjects:
        for session in sessions:
            subject_dir = data_path / subject / "ses-forrestgump" / "func"
            file_pattern = file_pattern_template.format(session)
            matched_files = list(subject_dir.rglob(file_pattern))
            # Assuming exactly one match per subject-session
            if len(matched_files) != 1:
                if verbose:
                    print(f"Warning: expected one file, found {len(matched_files)} for subject {subject}, session {session}")
                continue
            nii_path = matched_files[0]
            img = nib.load(str(nii_path))
            shape = img.shape  # (x, y, z, t)
            t = shape[-1]
            num_timepoints.append(t)
            all_session_paths.append((subject, session, nii_path))

    # Compute cumulative timepoints for offset alignment
    cumulative_timepoints = np.cumsum([0] + num_timepoints)

    # Open HDF5 for writing
    output_file = os.path.join(cfg.data.data_path, "data.hdf5") # specify in cfg: cfg.data.output_h5
    with h5py.File(output_file, 'w') as h5f:
        # We'll store each session as: /data/sessionXXX/{image, emotion, offset}
        # session index will follow the order of all_session_paths
        for global_session_idx, (subject, session, nii_path) in enumerate(all_session_paths):
            # Load image data for this session
            if verbose:
                print(f"Loading {nii_path} for subject={subject}, session={session}")

            img = nib.load(str(nii_path))
            img_data = img.get_fdata()  # Numpy array
            if normalization:
                img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-5)
            # Convert to float32 to save space
            img_data = img_data.astype(np.float32)

            # Session index and offsets
            global_session_idx = global_session_idx % 8 # 8 sessions per subject
            current_session_offset = session_offsets[global_session_idx]

            # Compute session_start and session_end in the original timeline (without added offset)
            session_start = cumulative_timepoints[global_session_idx] * 2  # TR=2s
            session_end = session_start + num_timepoints[global_session_idx] * 2

            if verbose:
                print(f"Session {global_session_idx}: subject={subject}, session={session}, "
                      f"session_start={session_start}, session_end={session_end}, offset={current_session_offset}")

            # Filter labels for this session
            session_labels = observer_labels[
                (observer_labels['offset'] >= session_start) &
                (observer_labels['offset'] < session_end)
            ].copy()

            # Filter out 'NONE' labels
            session_labels = session_labels[session_labels['emotion'] != 'NONE'].reset_index(drop=True)

            # Map emotions to indices
            session_labels['emotion_idx'] = session_labels['emotion'].map(emotion_idx)

            # Now we have aligned labels with offsets applied and no 'NONE'.
            # We need to store them. However, note that not every timepoint has a label now.
            # The dataset code returns data slice by indexing. Here, we'll store only the aligned offsets and emotions.

            # Extract arrays
            emotions_array = session_labels['emotion_idx'].to_numpy(dtype=np.int32)
            offsets_array = session_labels['offset'].to_numpy(dtype=np.float32)  # offset in seconds, float32 to save space

            # Create a group for this session
            group_name = f"data/{subject}_session{session}"
            grp = h5f.create_group(group_name)
            # Store image
            grp.create_dataset("image", data=img_data, compression="gzip", compression_opts=4)
            # Store emotions and offsets
            grp.create_dataset("emotion", data=emotions_array, compression="gzip", compression_opts=4)
            grp.create_dataset("offset", data=offsets_array, compression="gzip", compression_opts=4)

            # Print current offset, session, and subject
            # Current offset: we have multiple offsets. Print a summary (e.g., min and max)
            if len(offsets_array) > 0:
                print(f"Subject={subject}, Session={session}, Offset Range=({offsets_array.min()}, {offsets_array.max()})")
            else:
                print(f"Subject={subject}, Session={session}, no non-NONE labels")

            # Clear RAM
            del img_data, img, session_labels, emotions_array, offsets_array

    print(f"HDF5 file written to {output_file}")

# Example usage with Hydra:
@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig):
    create_hdf5(cfg)

if __name__ == "__main__":
    main()
