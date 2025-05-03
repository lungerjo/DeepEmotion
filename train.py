import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import build, eval, loss as loss_utils, logging as log_utils, debug

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
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(cfg_dict, allow_val_change=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build.load_dataloaders(cfg, modules, timers)

    output_dim = len(cfg.data.classification_emotion_idx) if cfg.data.label_mode == "classification" else cfg.data.soft_classification_output_dim

    model = build.build_model(cfg, output_dim, modules, timers)
    model = build.move_model_to_device(model, device, cfg, timers)
    criterion = loss_utils.get_loss(cfg.loss)
    optimizer = build.setup_optimizer(model, cfg, modules, timers)
    build.ensure_save_directory(cfg.data.save_model_path, modules, cfg, timers)

    if cfg.verbose.train:
        print(f"[TRAIN] Starting training")
        print(f"[TRAIN] Batch size: {cfg.train.batch_size}")
        print(f"[TRAIN] Epochs: {cfg.train.epochs}")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data, labels = batch["data_tensor"], batch["label_tensor"]
            data = data.float().to(device)
            labels = labels.to(device)

            if data.dim() == 4:
                data = data.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)

            if cfg.data.label_mode == "classification":
                correct += (outputs.argmax(1) == labels).sum().item()
            else:
                pred_class = outputs.argmax(dim=1)
                true_class = labels.argmax(dim=1)
                correct += (pred_class == true_class).sum().item()

                if cfg.verbose.debug and batch_idx < 2:
                    debug.print_soft_label(outputs, labels, batch_idx, epoch)

        train_accuracy = correct / total

        if cfg.data.label_mode == "classification":
            val_accuracy = eval.evaluate_classification(model, val_loader, device)
        else:
            val_accuracy = eval.evaluate_soft_classification(model, val_loader, device)

        if cfg.verbose.train:
            print(f"[TRAIN] Epoch {epoch+1}: Loss={total_loss/total:.4f}, Accuracy={train_accuracy:.2%}, Val Accuracy={val_accuracy:.2%}")

        log_utils.log_metrics(wandb, epoch, total_loss / total, train_accuracy, val_accuracy)

    model_path = os.path.join(cfg.data.save_model_path, wandb.run.id)
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] Model saved at {model_path}")

if __name__ == "__main__":
    main()