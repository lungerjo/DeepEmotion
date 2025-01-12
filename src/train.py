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
        for batch in tqdm(train_dataloader):
            # Extract data and labels
            data, labels = batch["data_tensor"], batch["label_tensor"]
            data = data.float().to(device)  # Ensure data is float for model input
            labels = labels.long().to(device)  # Ensure labels are integers for CrossEntropyLoss

            # Forward pass
            output = model(data)

            # Log raw predictions if desired
            wandb.log({
                "labels": labels.detach().cpu().numpy(),
                "predictions": output.argmax(dim=1).detach().cpu().numpy()
            })
            
            # Calculate loss
            loss = criterion(output, labels)  # No need to one-hot encode labels

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            _, predictions = torch.max(output, dim=1)  # Get class predictions
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0) 

        # Calculate epoch metrics
        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        normalized_loss = total_loss / total_samples if total_samples > 0 else 0

        # Log epoch metrics to WandB
        wandb.log({
            "epoch_loss": normalized_loss,
            "epoch_accuracy": accuracy,
            "epoch_duration": epoch_duration,
        })
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_data, val_labels = batch["data_tensor"], batch["label_tensor"]
                val_data = data.float().to(device)  # Ensure data is float for model input
                val_labels = labels.long().to(device)  # Ensure labels are integers for CrossEntropyLoss
                
                val_output = model(val_batch)
                _, val_predictions = torch.max(val_output, dim=1)
                val_true_labels = val_labels.argmax(dim=1)
                
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
