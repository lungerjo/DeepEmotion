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

    dataset = CrossSubjectDataset(cfg)

    if cfg.verbose:
        print("Dataset indexing and alignment complete.")
        print(f"Number of files: {len(dataset.data_files)}")
        print(f"Number of valid samples: {len(dataset.index_mappings)}")
        print(f"Subjects: {dataset.subjects}")
        print(f"Sessions: {dataset.sessions}")

    data_files = dataset.data_files
    index_mappings = dataset.index_mappings
    aligned_labels = dataset.aligned_labels
    subjects = dataset.subjects
    sessions = dataset.sessions
    classification_emotion_idx = dataset.classification_emotion_idx
    n_files = len(data_files)
    t_max = max(dataset.num_timepoints)

    if n_files == 0:
        raise ValueError("No data files found.")

    # Get spatial dimensions
    if cfg.verbose:
        print("Determining spatial shape from the first file...")
    x, y, z, _ = nib.load(str(data_files[0])).shape

    # Determine file_to_subject/session mappings
    file_to_subject = []
    file_to_session = []
    for fpath in data_files:
        sub = next((s for s in subjects if s in fpath.parts), None)
        ses = next((ssn for ssn in sessions if ssn in fpath.name), None)
        file_to_subject.append(sub)
        file_to_session.append(ses)

    file_to_subject = np.array(file_to_subject, dtype=object)
    file_to_session = np.array(file_to_session, dtype=object)

    # Merge aligned_labels + index_mappings
    index_mappings_df = pd.DataFrame(index_mappings)
    aligned_labels_reset = aligned_labels.reset_index(drop=False).rename(columns={"index": "aligned_label_idx"})
    merged_labels = pd.merge(index_mappings_df, aligned_labels_reset, on="aligned_label_idx", how="left")

    merged_labels = merged_labels.rename(columns={
        "data_file_idx": "file_index",
        "row_idx": "row_index",
        "offset": "time_offset"
    })

    session_map = {s: i for i, s in enumerate(sessions)}
    merged_labels["session"] = merged_labels["file_index"].apply(lambda fi: file_to_session[fi])
    merged_labels["session_idx"] = merged_labels["session"].apply(lambda s: session_map[s])
    merged_labels["global_idx"] = np.arange(len(merged_labels))
    csv_str = merged_labels.to_csv(index=False, sep="\t")

    # Init Zarr store
    store = zarr.group(output_zarr_path, overwrite=True)
    store.attrs["aligned_labels"] = csv_str

    chunk_size = (1, x, y, z, 1)
    data_zarr = store.create_dataset(
        "data",
        shape=(n_files, x, y, z, t_max),
        chunks=chunk_size,
        dtype="float32",
        compressor=None
    )

    if cfg.verbose:
        print(f"Writing data to Zarr dataset ({n_files} files)...")

    valid_timepoints = np.zeros(n_files, dtype="int32")
    for i, file_path in enumerate(data_files):
        if cfg.verbose.debug:
            print(f"[DEBUG] Loading file {i}: {file_path}")

        nii_img = nib.load(str(file_path))
        volume = nii_img.get_fdata(dtype=np.float32)
        t_current = volume.shape[-1]
        valid_timepoints[i] = t_current

        if cfg.data.normalization:
            volume = (volume - volume.mean()) / (volume.std() + 1e-5)

        data_zarr[i, ..., :t_current] = volume
        del volume, nii_img

        if cfg.verbose.debug:
            print(f"[DEBUG] Wrote volume {i} (t={t_current})")

    # Metadata arrays
    file_paths = np.array([str(p) for p in data_files], dtype=object)

    store.create_dataset("valid_timepoints", data=valid_timepoints, shape=(n_files,), dtype="int32")
    store.create_dataset("file_paths", data=file_paths.astype(f'U{max(len(p) for p in file_paths)}'), shape=(n_files,))
    store.create_dataset("subject_ids", data=np.array(subjects, dtype=f'U{max(len(s) for s in subjects)}'), shape=(len(subjects),))
    store.create_dataset("session_ids", data=np.array(sessions, dtype=f'U{max(len(s) for s in sessions)}'), shape=(len(sessions),))
    store.create_dataset("file_to_subject", data=file_to_subject.astype(f'U{max(len(s) for s in file_to_subject)}'), shape=(n_files,))
    store.create_dataset("file_to_session", data=file_to_session.astype(f'U{max(len(s) for s in file_to_session)}'), shape=(n_files,))
    store.attrs["emotions"] = list(classification_emotion_idx.keys())

    if cfg.verbose:
        print("Creating classification label array...")

    label_array = store.create_dataset(
        "classification_labels",
        shape=(n_files, t_max),
        chunks=(1, 50),
        dtype="int32",
        fill_value=-1
    )

    for m in index_mappings:
        label_array[m["data_file_idx"], m["row_idx"]] = m["label_idx"]

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
            f_idx = m["data_file_idx"]
            t_idx = m["row_idx"]
            offset = dataset.aligned_labels.iloc[m["aligned_label_idx"]]["offset"]
            reg_row = reg_df[reg_df["offset"] == offset]

            if not reg_row.empty:
                values = reg_row.iloc[0][soft_cols].values.astype(np.float32)
                soft_array[f_idx, t_idx, :] = values

                if cfg.verbose.debug:
                    print(f"[DEBUG] Soft label written at file={f_idx}, t={t_idx}")
            elif cfg.verbose.debug:
                print(f"[DEBUG] No match for soft label at offset={offset}")

    if cfg.verbose:
        print(f"Zarr dataset successfully written to: {output_zarr_path}")





@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:

    if cfg.verbose:
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        print(f"[DEBUG] Hydra project root: {cfg.project_root}")
        print(f"[DEBUG] Config-resolved data path: {cfg.data.data_path}")
        print(f"[DEBUG] Full resolved data path: {(Path(cfg.project_root) / cfg.data.data_path).resolve()}")
    output_path = str((Path(cfg.data.zarr_path)).resolve())
    if cfg.verbose:
        print("Starting Zarr dataset creation...")
    write_zarr_dataset(cfg, output_path)
    if cfg.verbose:
        print("Zarr dataset creation completed.")

if __name__ == "__main__":
    main()
