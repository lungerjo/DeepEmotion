from utils.dataset import get_data_loaders
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import LogisticRegressionModel, Small3DCNNClassifier
import time
import wandb
from collections import Counter
from tqdm import tqdm

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains a logistic regression model.
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="DeepEmotion", config=cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print("Loading dataloader...")
        
    train_dataloader, val_dataloader = get_data_loaders(cfg)
    print(f"Loaded Observations: {len(train_dataloader.dataset) + len(val_dataloader.dataset)}")
    """
    emotion_idx_inverse = {v: k for k, v in cfg.data.emotion_idx.items()}
    total_label_counter = Counter(
        label for dataloader in [train_dataloader, val_dataloader]
        for _, one_hot_label in dataloader
        for label in [emotion_idx_inverse[idx.item()] for idx in torch.argmax(one_hot_label, dim=1)]
    )
    print(dict(total_label_counter))
    """

    output_dim = len(cfg.data.emotion_idx)
    model = Small3DCNNClassifier(output_dim=output_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=cfg.data.weight_decay)

    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        model.train()
        for batch, label in tqdm(train_dataloader):
            
            batch, label = batch.float().to(device), label.float().to(device)

            output = model(batch)

            wandb.log({
                "labels": label.detach().cpu().numpy()
            })
            
            none_mask = (label.argmax(dim=1) != 0)  # Create a mask where 'NONE' labels (index 0) are excluded
            filtered_output = output[none_mask]
            filtered_label = label[none_mask]

            if filtered_output.numel() > 0:  # Check if there's any data left after filtering
                loss = criterion(filtered_output, filtered_label.argmax(dim=1))
            else:
                loss = torch.tensor(0.0, requires_grad=True) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predictions = torch.max(output, dim=1)
            true_labels = label.argmax(dim=1)
            none_mask = (label.argmax(dim=1) != 0)
            correct_predictions += (predictions == true_labels).sum().item()
            total_samples += true_labels.size(0)  # Count all samples since "NONE" labels are already removed

        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0  # Handle potential divide by zero
        normalized_loss = total_loss / total_samples if total_samples > 0 else 0

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch, val_label in val_dataloader:
                val_batch, val_label = val_batch.float().to(device), val_label.float().to(device)
                
                val_output = model(val_batch)
                _, val_predictions = torch.max(val_output, dim=1)
                val_true_labels = val_label.argmax(dim=1)
                
                val_correct += (val_predictions == val_true_labels).sum().item()
                val_total += val_true_labels.size(0)

        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
        f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds",
        f"Validation Accuracy: {val_accuracy * 100:.2f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": normalized_loss,
            "train_accuracy": accuracy,
            "val_accuracy": val_accuracy
        })
    
if __name__ == "__main__":
    main()
