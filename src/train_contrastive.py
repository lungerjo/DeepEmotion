# train_contrastive.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dutils.data import DataLoader, random_split
from tqdm import tqdm
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from utils.dataset import ContrastivePairDataset, ZarrDataset
from models.CNN import CNN

##################################################################
# 2. Contrastive Loss (Triplet)
##################################################################
# You can also use torch.nn.TripletMarginLoss(margin=1.0), or
# define your own with cos sim if you prefer (e.g., NT-Xent).
def contrastive_triplet_loss(anchor_embed, positive_embed, negative_embed, margin=1.0):
    """Simple triplet margin loss."""
    # Normalize embeddings (optional, but often done in contrastive learning)
    anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
    positive_norm = F.normalize(positive_embed, p=2, dim=1)
    negative_norm = F.normalize(negative_embed, p=2, dim=1)

    # L2-based distance
    pos_dist = (anchor_norm - positive_norm).pow(2).sum(1)
    neg_dist = (anchor_norm - negative_norm).pow(2).sum(1)

    # Triplet margin loss: max(0, margin + pos_dist - neg_dist)
    losses = F.relu(margin + pos_dist - neg_dist)
    return losses.mean()

##################################################################
# 3. Main Contrastive Training
##################################################################
@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Contrastive pretraining pipeline:
    1. Load base dataset (ZarrDataset).
    2. Wrap with ContrastivePairDataset.
    3. Train CNN with triplet margin loss.
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb:
        wandb.init(project="DeepEmotion-Contrastive", config=cfg_dict)
        wandb.config.update(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Loading dataset from {cfg.data.zarr_path}")

    # ----------------------------------------------------------------
    # Load Base Dataset
    # ----------------------------------------------------------------
    if cfg.verbose:
        print("[INFO] Initializing base dataset...")
    base_dataset = ZarrDataset(cfg.data.zarr_path)
    if cfg.verbose:
        print("[INFO] Wrapping dataset with contrastive pair generator...")


    # ----------------------------------------------------------------
    # Wrap with ContrastivePairDataset
    # ----------------------------------------------------------------
    contrastive_dataset = ContrastivePairDataset(base_dataset, seed=cfg.data.seed)

    if cfg.verbose:
        print(f"[INFO] Total samples: {len(contrastive_dataset)}")
        print("[INFO] Splitting dataset...")

    # Train-validation split
    train_ratio = cfg.train.train_ratio
    train_size = int(train_ratio * len(contrastive_dataset))
    val_size = len(contrastive_dataset) - train_size
    train_dataset, val_dataset = random_split(
        contrastive_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.data.seed)
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        drop_last=True  # ensures we always get full batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        drop_last=True
    )

    if cfg.verbose:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # ----------------------------------------------------------------
    # Model: Use an embedding dimension instead of output classes
    #        for contrastive pretraining
    # ----------------------------------------------------------------
    model = CNN(cfg=cfg, output_dim=128).to(device)

    # If you have a ResNet or other model:
    # model = ResNet(BasicBlock, [1,1,1,1], in_channels=1, num_classes=embedding_dim).to(device)

    # ----------------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.data.learning_rate,
        weight_decay=cfg.data.weight_decay
    )

    # You can also directly use torch.nn.TripletMarginLoss if you prefer:
    # triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Logging setup
    save_dir = cfg.data.save_model_path
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    # ----------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------
    for epoch in range(cfg.train.epochs):
        if cfg.verbose:
            print(f"\n[Epoch {epoch+1}] Starting training loop...")

        model.train()
        total_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}]")):
            if cfg.verbose and i % 10 == 0:
                print(f"[Epoch {epoch+1}] Training batch {i + 1}/{len(train_loader)}")

            anchor = batch["anchor"].float().to(device)
            positive = batch["positive"].float().to(device)
            negative = batch["negative"].float().to(device)

            if cfg.verbose and i == 0:
                print(f"[Epoch {epoch+1}] Sample shape: {anchor.shape}")

            anchor_embed = model(anchor, return_hidden=False)
            positive_embed = model(positive, return_hidden=False)
            negative_embed = model(negative, return_hidden=False)

            loss = contrastive_triplet_loss(anchor_embed, positive_embed, negative_embed, margin=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        if cfg.verbose:
            print(f"[Epoch {epoch+1}] Finished training. Avg loss: {avg_train_loss:.4f}")
            print("[INFO] Starting validation...")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if cfg.verbose and j % 10 == 0:
                    print(f"[Epoch {epoch+1}] Validating batch {j + 1}/{len(val_loader)}")

                anchor = batch["anchor"].float().to(device)
                positive = batch["positive"].float().to(device)
                negative = batch["negative"].float().to(device)

                anchor_embed = model(anchor, return_hidden=False)
                positive_embed = model(positive, return_hidden=False)
                negative_embed = model(negative, return_hidden=False)

                loss = contrastive_triplet_loss(anchor_embed, positive_embed, negative_embed, margin=1.0)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if cfg.verbose:
            print(f"[Epoch {epoch+1}] Validation complete. Avg val loss: {avg_val_loss:.4f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, f"contrastive_best_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            if cfg.verbose:
                print(f"[Epoch {epoch+1}] New best model saved to {save_path}")

    # ----------------------------------------------------------------
    # Final save
    # ----------------------------------------------------------------
    final_path = os.path.join(save_dir, "contrastive_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Contrastive training finished. Final model saved at {final_path}")

if __name__ == "__main__":
    main()
