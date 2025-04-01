import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import hydra
import os
from models.CNN import CNN
from omegaconf import DictConfig, OmegaConf
import wandb
from utils.dataset import get_data_loaders
from collections import Counter
import tqdm
import time

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def load_pretrained_model(cfg, model_path, num_original_classes):
    """Load a pretrained model"""
    model = CNN(cfg, num_original_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def create_transfer_model(pretrained_model, num_classes, freeze_features=True):
    """Create a transfer model by modifying the last layer of the pretrained model"""
    model = copy.deepcopy(pretrained_model)
    model.fc2 = nn.Linear(model.fc2.in_features, num_classes)
    if freeze_features:
        for name, param in model.named_parameters():
            if 'fc2' not in name:
                param.requires_grad = False
    return model

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def train_fewshot(cfg: DictConfig) -> None:
    """
    Main function for transfer learning with a pretrained 3D CNN model.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb:
        wandb.init(project="DeepEmotion-Transfer", config=cfg_dict)
        wandb.config.update(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Loading dataloader from {cfg.data.zarr_path}")
        
    train_dataloader, val_dataloader = get_data_loaders(cfg)
    print(f"Loaded Observations: {len(train_dataloader.dataset) + len(val_dataloader.dataset)}")
    
    # Number of classes for new emotions
    output_dim = len(cfg.data.emotion_idx)

    if cfg.train.print_label_frequencies: 
        def get_label_frequencies(dataloader):
            label_counts = Counter()
            for batch in dataloader: 
                data, labels = batch["data_tensor"], batch["label_tensor"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                label_counts.update(labels.flatten()) 
            return label_counts

        train_label_counts = get_label_frequencies(train_dataloader)
        val_label_counts = get_label_frequencies(val_dataloader)
        total_label_counts = train_label_counts + val_label_counts
        for label, count in sorted(total_label_counts.items()):
            inverse_emotion_idx = {v: k for k, v in cfg.data.emotion_idx.items()}
            emotion_name = inverse_emotion_idx[label] 
            print(f"{emotion_name}: {count}")
    
    # Load the pretrained model
    if not hasattr(cfg, 'transfer') or not hasattr(cfg.transfer, 'pretrained_model_path'):
        raise ValueError("Pretrained model path must be specified for transfer learning (cfg.transfer.pretrained_model_path)")
    
    print(f"Loading pretrained model from {cfg.transfer.pretrained_model_path}")
    

    pretrained_model = CNN(cfg=cfg, output_dim=5)
    
    # Load the pretrained weights
    pretrained_model.load_state_dict(torch.load(cfg.transfer.pretrained_model_path, 
                                              map_location=device))
    
    # Set default freeze_features if not specified
    freeze_features = True
    if hasattr(cfg, 'transfer') and hasattr(cfg.transfer, 'freeze_features'):
        freeze_features = cfg.transfer.freeze_features
    
    # Create the transfer model with new output dimension
    model = create_transfer_model(
        pretrained_model, 
        output_dim, 
        freeze_features=freeze_features
    )
    
    # Set up training with possibly fewer trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.verbose:
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
    # Set up optimizer with only trainable parameters
    optimizer = optim.Adam(
        trainable_params, 
        lr=cfg.data.learning_rate, 
        weight_decay=cfg.data.weight_decay
    )
    
    # Standard training loop for fine-tuning
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Directory to save model
    save_dir = cfg.data.save_model_path
    os.makedirs(save_dir, exist_ok=True) if not os.path.exists(save_dir) else None
    
    best_val_accuracy = 0.0
    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        model.train()
        for batch in tqdm(train_dataloader):
            data, labels = batch["data_tensor"], batch["label_tensor"]

            data = data.float().to(device)
            labels = labels.long().to(device)

            output = model(data)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predictions = torch.max(output, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0) 
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        normalized_loss = total_loss / total_samples if total_samples > 0 else 0
        
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_data, val_labels = val_batch["data_tensor"], val_batch["label_tensor"]
                val_data = val_data.float().to(device)
                val_labels = val_labels.long().to(device)

                val_output = model(val_data)
                _, val_predictions = torch.max(val_output, dim=1)
                val_correct += (val_predictions == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
            f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds, "
            f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if cfg.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": normalized_loss,
                "train_accuracy": accuracy,
                "val_accuracy": val_accuracy
            })
            
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if cfg.wandb:
                model_path_torch = os.path.join(save_dir, f"{wandb.run.id}-transfer-best.pth")
            else:
                model_path_torch = os.path.join(save_dir, "transfer-best.pth")
            torch.save(model.state_dict(), model_path_torch)
            print(f"New best model saved with validation accuracy: {val_accuracy * 100:.2f}%")

    # Save the final model
    if cfg.wandb:
        model_path_torch = os.path.join(save_dir, f"{wandb.run.id}-transfer-final.pth")
    else:
        model_path_torch = os.path.join(save_dir, "transfer-final.pth")
    torch.save(model.state_dict(), model_path_torch)
    print(f"Final model saved at {model_path_torch}")
    

if __name__ == "__main__":
    train_fewshot()