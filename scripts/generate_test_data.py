# generate_synthetic_test_data.py

import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import hydra
from pathlib import Path
import shutil  # Import shutil to handle directory removal

# Get config
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Adjust as needed
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
CONFIG_PATH = str(Path(PROJECT_ROOT) / "src" / "configs")

@hydra.main(config_path=CONFIG_PATH, config_name="test", version_base=None)
def generate_synthetic_test_data(cfg: DictConfig):

    raw_test_data_dir = Path(cfg.paths.raw_test_path)
    annotations_test_data_dir = Path(cfg.paths.annotations_test_path)
    
    # Ensure directories exist and overwrite any existing data
    # Remove directories if they already exist
    if raw_test_data_dir.exists() and raw_test_data_dir.is_dir():
        shutil.rmtree(raw_test_data_dir)
    if annotations_test_data_dir.exists() and annotations_test_data_dir.is_dir():
        shutil.rmtree(annotations_test_data_dir)

    # Recreate directories
    raw_test_data_dir.mkdir(parents=True, exist_ok=True)
    annotations_test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define number of subjects, runs, features, and classes
    num_subjects = cfg.synthetic_data.num_subjects
    runs_per_subject = cfg.synthetic_data.runs_per_subject
    num_features = cfg.synthetic_data.num_features
    num_classes = cfg.synthetic_data.num_classes
    labels = cfg.synthetic_data.labels

    annotations = []

    for subject_num in range(1, num_subjects + 1):
        subject_id = f'sub-{subject_num:02d}'
        subject_dir = raw_test_data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        for run_num in range(1, runs_per_subject + 1):
            # Randomly select a class index
            class_index = np.random.randint(0, num_classes)
            label = labels[class_index]

            # Generate synthetic data from N(class_index, 1)
            data = np.random.normal(
                loc=class_index, 
                scale=1.0, 
                size=(100, num_features)  # 100 timepoints, num_features features
            )

            fmri_filename = f'{subject_id}_task-avmovie_run-{run_num}_bold.csv'
            fmri_filepath = subject_dir / fmri_filename

            # Save data to CSV
            pd.DataFrame(data).to_csv(fmri_filepath, index=False)

            # Append annotation entry
            annotations.append({
                'subject': subject_id,
                'run': run_num,
                'label': label
            })

            # Print out the path that is being written to
            print(f"Data written to: {fmri_filepath}")

    # Save annotations to TSV file
    annotations_df = pd.DataFrame(annotations)
    annotation_filename = cfg.synthetic_data.annotation_filename
    annotation_filepath = annotations_test_data_dir / annotation_filename
    annotations_df.to_csv(annotation_filepath, sep='\t', index=False)

    # Print out the path of the annotation file
    print(f"Annotations written to: {annotation_filepath}")

    print(f"Synthetic test data generation complete.")
    print(f"Data saved in '{raw_test_data_dir}'")
    print(f"Annotations saved in '{annotation_filepath}'")

if __name__ == '__main__':
    generate_synthetic_test_data()
