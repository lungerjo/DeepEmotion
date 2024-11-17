from utils.dataset import get_data_loaders
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import DeepLogisticRegressionModel
import time
import wandb


@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains a logistic regression model.
    """

    # Initialize logging (assuming W&B here)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="DeepEmotion", config=cfg_dict)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print("Loading dataloader...")
        
    train_dataloader, val_dataloader = get_data_loaders(cfg)

    # Set up model, loss function, optimizer
    input_dim = 132 * 175 * 48  # Flattened input size
    output_dim = len(cfg.data.emotion_idx)  # Number of classes from emotion index
    model = DeepLogisticRegressionModel(input_dim, output_dim)  # Multi-class logistic regression
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=cfg.data.weight_decay)

    # Training loop
    for epoch in range(cfg.train.epochs):  # Number of training epochs from config
        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Training step
        model.train()  # Set model to training mode
        for batch, label in train_dataloader:
            # Move batch and label to device
            flattened = batch.flatten(start_dim=1).float().to(device)
            label = label.to(device)

            # Forward pass
            output, hidden_activations = model(flattened)

            wandb.log({
                "hidden_activations": hidden_activations.detach().cpu().numpy(),
                "labels": label.detach().cpu().numpy()
            })
            
            # Compute loss
            loss = criterion(output, label.argmax(dim=1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Convert the output logits to predicted classes
            _, predictions = torch.max(output, dim=1)
            true_labels = label.argmax(dim=1)
            
            # Calculate number of correct predictions
            correct_predictions += (predictions == true_labels).sum().item()
            total_samples += label.size(0)

        # Calculate training metrics
        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = correct_predictions / total_samples
        normalized_loss = total_loss / total_samples

        # Validation step
        model.eval()  # Set model to evaluation mode
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # No need to compute gradients for validation
            for val_batch, val_label in val_dataloader:
                val_flattened = val_batch.flatten(start_dim=1).float().to(device)
                val_label = val_label.to(device)
                
                # Forward pass
                val_output, _ = model(val_flattened)
                _, val_predictions = torch.max(val_output, dim=1)
                val_true_labels = val_label.argmax(dim=1)
                
                # Calculate validation accuracy
                val_correct += (val_predictions == val_true_labels).sum().item()
                val_total += val_label.size(0)

        val_accuracy = val_correct / val_total
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
        f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds",
        f"Validation Accuracy: {val_accuracy * 100:.2f}%\n")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": normalized_loss,
            "train_accuracy": accuracy,
            "val_accuracy": val_accuracy
        })
    

if __name__ == "__main__":
    main()
