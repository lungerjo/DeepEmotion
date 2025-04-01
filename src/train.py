from utils.dataset import get_data_loaders
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from models.CNN import CNN
from models.resnet import ResNet, BasicBlock
import time
import wandb
import pickle
from collections import Counter
from tqdm import tqdm

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains a logistic regression model.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb:
        wandb.init(project="DeepEmotion", config=cfg_dict)
        wandb.config.update(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Loading dataloader from {cfg.data.zarr_path}")
        
    train_dataloader, val_dataloader = get_data_loaders(cfg)
    print(f"Loaded Observations: {len(train_dataloader.dataset) + len(val_dataloader.dataset)}")
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

    if cfg.data.load_model:
        model_path_torch = cfg.data.load_model_path
        print(f"Loading the model from {model_path_torch}...")
        state_dict_torch = torch.load(model_path_torch, weights_only=True)
        model.load_state_dict(state_dict_torch)
        print(f"Loaded the model from {model_path_torch}.")
    elif cfg.model == "CNN":
        model = CNN(cfg=cfg, output_dim=output_dim)
    elif cfg.model == "ResNet":
        model = ResNet(BasicBlock, [1, 1, 1, 1], in_channels=1, num_classes=22)
        def initialize_new_layers(model):
            for name, module in model.named_modules():
                if 'fc' in name:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        initialize_new_layers(model)
    else:
        raise ValueError(f"Error: load model as cfg.data.load_model = <model_path> or initialize valid model for cfg.model")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.data.learning_rate, weight_decay=cfg.data.weight_decay)
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

            data = data.float().to(device)  # Ensure data is float for model input
            labels = labels.long().to(device)  # Ensure labels are integers for CrossEntropyLoss
            if data.dim() == 4:
                data = data.unsqueeze(1)

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
                val_data = val_data.float().to(device)  # Ensure data is float for model input
                val_labels = val_labels.long().to(device)  # Ensure labels are integers for CrossEntropyLoss
                if val_data.dim() == 4:
                    val_data = val_data.unsqueeze(1)

                val_output = model(val_data)
                _, val_predictions = torch.max(val_output, dim=1)
                val_correct += (val_predictions == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
        f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds",
        f"Validation Accuracy: {val_accuracy * 100:.2f}")

        if cfg.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": normalized_loss,
                "train_accuracy": accuracy,
                "val_accuracy": val_accuracy
            })


    if cfg.wandb:
        model_path_torch = os.path.join(save_dir, f"{wandb.run.id}-sub02-20.pth")
    else:
        model_path_torch = os.path.join(save_dir, "model-sub02-20.pth")
    torch.save(model.state_dict(), model_path_torch)
    print(f"Model saved at {save_dir}")
    

if __name__ == "__main__":
    main()
