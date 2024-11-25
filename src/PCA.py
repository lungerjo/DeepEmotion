from utils.dataset import get_data_loader
import hydra
from omegaconf import DictConfig
import torch
from sklearn.decomposition import PCA
import csv
import os


@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset, and
    writes PCA results to a file specified by the configuration.

    Args:
    - cfg: The configuration object loaded by Hydra.
    """
    print(cfg.project_root)
    if cfg.verbose:
        print("Loading dataloader")
        
    dataloader = get_data_loader(cfg)

    # Set up PCA
    pca = PCA(n_components=3)  # Adjust the number of components as needed

    # Iterate over subjects and runs in the configuration
    for subject in cfg.data.subjects:
        for session in cfg.data.sessions:
            # Construct the file path from the configuration
            output_file_path = os.path.join(
                cfg.project_root,
                "data",
                "derivative",
                "PCA_local",
                f"{subject}_{session}.csv"
            )
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Open a CSV file for writing
            with open(output_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Iterate over the dataloader
                for batch, path in dataloader:
                    # Each batch has size [1, 1, 132, 175, 48]
                    # We will first remove the batch and channel dimensions
                    batch = batch.squeeze(0).squeeze(0)  # Shape [132, 175, 48]
                    
                    # Flatten to 2D array (for PCA input)
                    flattened = batch.flatten(start_dim=1)  # Shape [132, 175*48]
                    
                    # Apply PCA
                    pca_result = pca.fit_transform(flattened)  # Result will be [132, n_components]
                    
                    # Write the result row by row to the CSV
                    for row in pca_result:
                        writer.writerow(row.tolist())
                
            print(f"PCA results written to {output_file_path}")


if __name__ == "__main__":
    main()