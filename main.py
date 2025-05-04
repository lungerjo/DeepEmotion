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
    np = modules["np"]
    pd = modules["pd"]

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.wandb:
        build.setup_wandb(cfg, cfg_dict, timers)
        wandb.config.update(cfg_dict, allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = (
        len(cfg.data.classification_emotion_idx)
        if cfg.data.label_mode == "classification"
        else cfg.data.soft_classification_output_dim
    )
    model = build.build_model(cfg, output_dim, modules, timers)
    model = build.move_model_to_device(model, device, cfg, timers)

    if cfg.data.load_model and cfg.data.load_model_path:
        print("Loading weights from:", cfg.data.load_model_path)
    model.eval() if cfg.mode in ["eval", "PCA"] else model.train()

    if cfg.mode == "train":
        if cfg.verbose.train:
            print(f"[TRAIN] batch_size: {cfg.train.batch_size}")
            print(f"[TRAIN] epochs: {cfg.train.epochs}")
            print(f"[TRAIN] loss: {cfg.loss}")
            print(f"[TRAIN] model: {cfg.model}")
        criterion = loss_utils.get_loss(cfg)
        optimizer = build.setup_optimizer(model, cfg, modules, timers)
        build.ensure_save_directory(cfg.data.save_model_path, modules, cfg, timers)
        train_loader, val_loader = build.load_dataloaders(cfg, modules, timers)

        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                data = batch["data_tensor"].float().to(device)
                labels = batch["label_tensor"].to(device)
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
                    acc = eval.compute_masked_soft_accuracy(outputs, labels, none_index=-1)
                    correct += acc * labels.size(0)  # weight by batch size

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

    elif cfg.mode == "eval":
        _, val_loader = build.load_dataloaders(cfg, modules, timers)
        if cfg.data.label_mode == "classification":
            val_accuracy = eval.evaluate_classification(model, val_loader, device)
        else:
            val_accuracy = eval.evaluate_soft_classification(model, val_loader, device)
        print(f"[EVAL] Validation Accuracy: {val_accuracy:.2%}")

    elif cfg.mode == "PCA":
        from src.utils.dataset import ZarrDataset
        from torch.utils.data import DataLoader

        ds = ZarrDataset(
            zarr_path=cfg.data.zarr_path,
            label_mode=cfg.data.label_mode,
            debug=cfg.verbose.build,
            cfg=cfg,
        )

        allowed = set(cfg.data.subjects)
        idxs = ds.valid_indices[:, 0]
        subs = np.array(ds.file_to_subject)[idxs]
        mask = np.isin(subs, list(allowed))
        ds.valid_indices = ds.valid_indices[mask]

        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=getattr(cfg.train, "num_workers", 0))
        hidden, labels = [], []
        meta = {"global_idx": [], "subject": [], "session": []}
        correct, total = 0, 0

        for batch in loader:
            x = batch["data_tensor"].float().to(device)
            y = batch["label_tensor"].to(device)
            if x.dim() == 4:
                x = x.unsqueeze(1)
            with torch.no_grad():
                logits, h = model(x, return_hidden=True)
            pred = logits.argmax(dim=1)
            target = y.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += y.size(0)
            h_np = h.cpu().numpy()
            lbl_np = y.cpu().numpy()
            for i in range(h_np.shape[0]):
                hidden.append(h_np[i])
                labels.append(lbl_np[i])
                meta["global_idx"].append(batch["global_idx"][i].item())
                meta["subject"].append(batch["subject"][i])
                meta["session"].append(batch["session"][i])

        accuracy = correct / total
        print(f"[PCA] Soft classification accuracy: {accuracy:.2%}")
        H = np.vstack(hidden)
        pca = modules["sklearn"].decomposition.PCA(n_components=2)
        pcs = pca.fit_transform(H)

        soft_cols = [c for c in ds.aligned_labels.columns if c.startswith("e_")]
        df_meta = pd.DataFrame(meta)
        df_meta["PCA1"], df_meta["PCA2"] = pcs[:, 0], pcs[:, 1]
        df_soft = pd.DataFrame(labels, columns=soft_cols)
        df = pd.concat([df_meta, df_soft], axis=1)
        df.to_csv(cfg.data.pca_out_path, index=False)
        print(f"[PCA] Saved output to {cfg.data.pca_out_path}")

    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

if __name__ == "__main__":
    main()
