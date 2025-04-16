import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import build

@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    timers = {}
    modules = build.imports(cfg, timers)

    torch = modules["torch"]
    wandb = modules["wandb"]
    os = modules["os"]
    tqdm = modules["tqdm"]

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb:
        build.setup_wandb(cfg, cfg_dict, timers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build.load_dataloaders(cfg, modules, timers)

    output_dim = len(cfg.data.emotion_idx)
    model = build.build_model(cfg, output_dim, modules, timers)
    model = build.move_model_to_device(model, device, cfg, timers)
    criterion, optimizer = build.setup_optimizer_and_loss(model, cfg, modules, timers)
    build.ensure_save_directory(cfg.data.save_model_path, modules, cfg, timers)

    if cfg.verbose.train:
        print(f"[TRAIN] Starting training")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader):
            data, labels = batch["data_tensor"], batch["label_tensor"]
            data, labels = data.float().to(device), labels.long().to(device)
            if data.dim() == 4: data = data.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        def evaluate(model, val_loader, device):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    data, labels = batch["data_tensor"].to(device), batch["label_tensor"].to(device)
                    if data.dim() == 4:
                        data = data.unsqueeze(1)
                    preds = model(data).argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            return correct / total if total > 0 else 0
    
        val_acc = evaluate(model, val_loader, device)

        if cfg.verbose.train:
            print(f"[TRAIN] Epoch {epoch+1}: Loss={total_loss/total:.4f}, Accuracy={accuracy:.2%}, Val Accuracy={val_acc:.2%}")

        if cfg.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_loss / total,
                "train_accuracy": accuracy,
                "val_accuracy": val_acc
            })

    model_path = os.path.join(cfg.data.save_model_path, wandb.run.id)
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] Model saved at {model_path}")

if __name__ == "__main__":
    main()