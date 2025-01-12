from utils.dataset import get_data_loaders
from models.i3d import I3D
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
from torch.cuda.amp import autocast, GradScaler

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains the I3D model.
    """

    scaler = GradScaler()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="DeepEmotion", config=cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print("Loading dataloader...")

    # Ensure zarr_path is correctly set
    zarr_path = cfg.data.zarr_path
    if not zarr_path:
        raise ValueError("The zarr_path is not set in the configuration.")

    train_dataloader, val_dataloader = get_data_loaders(cfg)
    print(f"Loaded Observations: {len(train_dataloader.dataset) + len(val_dataloader.dataset)}")

    num_classes = len(cfg.data.emotion_idx)  # Calculate number of classes dynamically
    model = I3D(num_classes=num_classes)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_accuracy = 0.0

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

            with torch.amp.autocast('cuda'):
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                data = batch["data_tensor"].to(device)
                labels = batch["label_tensor"].to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, val_predictions = torch.max(outputs, dim=1)
                    val_correct += (val_predictions == labels).sum().item()
                    val_total += labels.size(0)

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
              f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": normalized_loss,
            "train_accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Save the model if the validation accuracy is the best we've seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"/home/paperspace/DeepEmotion/output/models/i3d_best.pth")

if __name__ == "__main__":
    main()