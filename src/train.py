from utils.dataset import get_data_loader
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader and processes the dataset.

    Args:
    - cfg: The configuration object loaded by Hydra.
    """

    if cfg.verbose: print("Loading dataloader")
    dataloader = get_data_loader(cfg, batch_size=64, shuffle=True)
    for data, path in dataloader:
        print(data.shape)

if __name__ == "__main__":
    main()