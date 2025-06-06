import zarr
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import hydra
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.dataset import StudyForrestVolumeIndexer

def write_zarr_dataset(cfg: DictConfig, output_zarr_path: str):
    dataset = StudyForrestVolumeIndexer(cfg)

    if cfg.verbose.build:
        print("[BUILD] Dataset indexing and alignment complete.")
        print(f"[BUILD] Number of volumes: {len(dataset.volume_paths)}")
        print(f"[BUILD] Number of valid samples: {len(dataset.index_mappings)}")
        print(f"[BUILD] Subjects: {dataset.subjects}")
        print(f"[BUILD] Sessions: {dataset.sessions}")

    volume_paths = dataset.volume_paths
    index_mappings = dataset.index_mappings
    labels_df = dataset.labels_df
    subjects = dataset.subjects
    sessions = dataset.sessions
    classification_emotion_idx = dataset.classification_emotion_idx
    n_files = len(volume_paths)
    t_max = max(dataset.volume_lengths)

    if n_files == 0:
        raise ValueError("No data files found.")

    # Get spatial dimensions
    if cfg.verbose.build:
        print("[BUILD] Determining spatial shape from the first file...")
    x, y, z, _ = nib.load(str(volume_paths[0])).shape

    # Determine file_to_subject/session mappings
    file_to_subject = []
    file_to_session = []
    for fpath in volume_paths:
        sub = next((s for s in subjects if s in fpath.parts), None)
        match = re.search(r'run-(\d+)', fpath.name)
        ses = match.group(1) if match else None
        file_to_subject.append(sub)
        file_to_session.append(ses)

    file_to_subject = np.array(file_to_subject, dtype=object)
    file_to_session = np.array(file_to_session, dtype=object)

    if cfg.verbose.debug:
        assert all(sub in subjects for sub in file_to_subject), \
            "[ASSERT FAILED] Unknown subject in file_to_subject"
        assert all(ses in sessions for ses in file_to_session), \
            "[ASSERT FAILED] Unknown session in file_to_session"

    # Step 1: Build index_mappings
    index_mappings_df = pd.DataFrame(index_mappings)

    # Step 2: Reset index of aligned_labels so it can be merged
    aligned_labels_reset = labels_df.reset_index(drop=False).rename(columns={"index": "label_idx"})

    # Step 3: Merge
    merged_labels = pd.merge(index_mappings_df, aligned_labels_reset, on="label_idx", how="left")

    # Step 5: Add metadata columns
    session_map = {s: i for i, s in enumerate(sessions)}
    merged_labels["session"] = merged_labels["vol_idx"].apply(lambda fi: file_to_session[fi])
    merged_labels["session_idx"] = merged_labels["session"].apply(lambda s: session_map[s])
    merged_labels["global_idx"] = np.arange(len(merged_labels))

    # Step 6: Create CSV string
    csv_str = merged_labels.to_csv(index=False, sep="\t")

    # Step 7: Init Zarr store and attach metadata
    store = zarr.group(output_zarr_path, overwrite=True)
    store.attrs["aligned_labels"] = csv_str

    # Step 8: Write valid_indices array
    valid_indices_array = merged_labels[["vol_idx", "local_idx"]].values.astype("int32")
    store.create_dataset("valid_indices", data=valid_indices_array, shape=valid_indices_array.shape)

    if cfg.verbose.debug:
        assert len(index_mappings_df) == len(merged_labels), \
            f"[ASSERT FAILED] merged_labels length ({len(merged_labels)}) != index_mappings_df length ({len(index_mappings_df)})"
        assert not merged_labels["global_idx"].isnull().any(), \
            "[ASSERT FAILED] Null global_idx values found in merged_labels"

    chunk_size = (1, x, y, z, 1)
    data_zarr = store.create_dataset(
        "data",
        shape=(n_files, x, y, z, t_max),
        chunks=chunk_size,
        dtype="float32",
        compressor=None
    )

    if cfg.verbose.build :
        print(f"[BUILD] Writing data to Zarr dataset ({n_files} files)...")

    valid_timepoints = np.zeros(n_files, dtype="int32")
    for i, file_path in enumerate(volume_paths):
        if cfg.verbose.debug:
            print(f"[BUILD] Loading file {i}: {file_path}")

        nii_img = nib.load(str(file_path))
        volume = nii_img.get_fdata(dtype=np.float32)
        t_current = volume.shape[-1]
        valid_timepoints[i] = t_current

        if cfg.verbose.debug:
            assert t_current <= t_max, f"[ASSERT FAILED] t_current={t_current} exceeds t_max={t_max}"

        data_zarr[i, ..., :t_current] = volume
        del volume, nii_img

    # Metadata arrays
    file_paths = np.array([str(p) for p in volume_paths], dtype=object)

    store.create_dataset("valid_timepoints", data=valid_timepoints, shape=(n_files,), dtype="int32")
    store.create_dataset("file_paths", data=file_paths.astype(f'U{max(len(p) for p in file_paths)}'), shape=(n_files,))
    store.create_dataset("subject_ids", data=np.array(subjects, dtype=f'U{max(len(s) for s in subjects)}'), shape=(len(subjects),))
    store.create_dataset("session_ids", data=np.array(sessions, dtype=f'U{max(len(s) for s in sessions)}'), shape=(len(sessions),))
    store.create_dataset("file_to_subject", data=file_to_subject.astype(f'U{max(len(s) for s in file_to_subject)}'), shape=(n_files,))
    store.create_dataset("file_to_session", data=file_to_session.astype(f'U{max(len(s) for s in file_to_session)}'), shape=(n_files,))
    store.attrs["emotions"] = list(classification_emotion_idx.keys())

    if cfg.verbose.build:
        print("[BUILD] Creating classification label array...")

    label_array = store.create_dataset(
        "classification_labels",
        shape=(n_files, t_max),
        chunks=(1, 50),
        dtype="int32",
        fill_value=-1
    )

    for m in index_mappings:
        f_idx = m["vol_idx"]
        t_idx = m["local_idx"]
        if cfg.verbose.debug:
            assert 0 <= f_idx < n_files, f"[ASSERT FAILED] file idx {f_idx} out of bounds"
            assert 0 <= t_idx < t_max, f"[ASSERT FAILED] row idx {t_idx} out of bounds (t_max={t_max})"
        label_array[f_idx, t_idx] = m["label_idx"]

    if cfg.data.get("soft_classification_label_path"):
        if cfg.verbose:
            print("Adding soft_classification_labels to Zarr...")

        reg_df = pd.read_csv(Path(cfg.data.soft_classification_label_path).expanduser(), sep="\t")
        soft_cols = [col for col in reg_df.columns if col != "offset"]
        num_soft = len(soft_cols)

        soft_array = store.create_dataset(
            "soft_classification_labels",
            shape=(n_files, t_max, num_soft),
            chunks=(1, 50, num_soft),
            dtype="float32",
            fill_value=np.nan
        )

        for m in dataset.index_mappings:
            if cfg.data.label_mode != "soft_classification":
                continue
            f_idx = m["vol_idx"]
            t_idx = m["local_idx"]
            offset = m["offset"]
            reg_row = reg_df[reg_df["offset"] == offset]

            if cfg.verbose.debug:
                assert not reg_row.empty, \
                    f"[ASSERT FAILED] No soft label match for offset {offset} (file={f_idx}, t={t_idx})"

            if not reg_row.empty:
                values = reg_row.iloc[0][soft_cols].values.astype(np.float32)
                soft_array[f_idx, t_idx, :] = values

                if cfg.verbose.debug:
                    print(f"[DEBUG] Soft label written at file={f_idx}, t={t_idx}")
            elif cfg.verbose.debug:
                print(f"[DEBUG] No match for soft label at offset={offset}")

    if cfg.verbose.build:
        print(f"[BUILD] Zarr dataset successfully written to: {output_zarr_path}")


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:

    if cfg.verbose.debug:
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        print(f"[DEBUG] Hydra project root: {cfg.project_root}")
        print(f"[DEBUG] Config-resolved data path: {cfg.data.data_path}")
        print(f"[DEBUG] Full resolved data path: {(Path(cfg.project_root) / cfg.data.data_path).resolve()}")
    output_path = str((Path(cfg.data.zarr_path)).resolve())
    if cfg.verbose.build:
        print("[BUILD] Starting Zarr dataset creation...")
    write_zarr_dataset(cfg, output_path)
    if cfg.verbose.build:
        print("[BUILD] Zarr dataset creation completed.")

if __name__ == "__main__":
    main()
