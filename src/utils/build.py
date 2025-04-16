import time
import sys
from types import SimpleNamespace
from src.utils.timer import time_step

def verbose_import(name, import_fn, cfg, timers=None):
    if cfg.verbose.imports:
        print(f"[IMPORT] Loading {name}...")
    start = time.time() if cfg.verbose.time else None
    result = import_fn()
    if cfg.verbose.time:
        duration = time.time() - start
        print(f"[TIMER] {name} took {duration:.2f} seconds")
        if timers is not None:
            timers[name] = duration
    return result

def imports(cfg, timers=None):
    modules = {}
    modules["torch"] = verbose_import("torch", lambda: __import__("torch"), cfg, timers)
    modules["nn"] = modules["torch"].nn
    modules["optim"] = modules["torch"].optim
    modules["os"] = verbose_import("os", lambda: __import__("os"), cfg, timers)
    modules["time"] = verbose_import("time", lambda: __import__("time"), cfg, timers)
    modules["wandb"] = verbose_import("wandb", lambda: __import__("wandb"), cfg, timers)
    modules["Counter"] = verbose_import("collections", lambda: __import__("collections"), cfg, timers).Counter
    modules["tqdm"] = verbose_import("tqdm", lambda: __import__("tqdm"), cfg, timers).tqdm
    modules["ZarrDataset"] = verbose_import(
    "utils.dataset",
    lambda: __import__("src.utils.dataset", fromlist=["ZarrDataset"]),
    cfg, timers
    ).ZarrDataset
    
    modules["CNN"] = verbose_import(
        "src.models.CNN",
        lambda: __import__("src.models.CNN", fromlist=["CNN"]),
        cfg, timers
    ).CNN

    resnet = verbose_import(
        "models.resnet",
        lambda: __import__("src.models.resnet", fromlist=["ResNet", "BasicBlock"]),
        cfg, timers
    )
    modules["ResNet"] = resnet.ResNet
    modules["BasicBlock"] = resnet.BasicBlock

    modules["time_step"] = verbose_import(
        "src.utils.timer.time_step",
        lambda: __import__("src.utils.timer", fromlist=["time_step"]),
        cfg, timers
    ).time_step

    return modules

def setup_wandb(cfg, cfg_dict, timers):
    if cfg.verbose.build:
        print(f"[BUILD] Initializing Weights & Biases...")
    @time_step("WandB Init", timers=timers, verbose=cfg.verbose.time)
    def _setup():
        import wandb
        wandb.init(project="DeepEmotion", config=cfg_dict)
        wandb.config.update(cfg_dict)
    _setup()

def load_dataloaders(cfg, timers):
    from utils.dataset import get_data_loaders
    @time_step("Dataloader Load", timers=timers, verbose=cfg.verbose.time)
    def _load():
        return get_data_loaders(cfg)
    return _load()

def build_model(cfg, output_dim, modules, timers):
    if cfg.verbose.build:
        print(f"[BUILD] Constructing model: {cfg.model}...")
    torch = modules["torch"]
    CNN = modules["CNN"]
    ResNet = modules["ResNet"]
    BasicBlock = modules["BasicBlock"]

    @time_step("Model Init", timers=timers, verbose=cfg.verbose.time)
    def _init():
        if cfg.data.load_model:
            model = CNN(cfg, output_dim) if cfg.model == "CNN" else ResNet(BasicBlock, [1, 1, 1, 1], in_channels=1, num_classes=22)
            model.load_state_dict(torch.load(cfg.data.load_model_path, weights_only=True))
            print(f"[BUILD] Loaded model from {cfg.data.load_model_path}")
        elif cfg.model == "CNN":
            model = CNN(cfg, output_dim)
        elif cfg.model == "ResNet":
            model = ResNet(BasicBlock, [1, 1, 1, 1], in_channels=1, num_classes=22)
            for name, module in model.named_modules():
                if 'fc' in name and hasattr(module, 'weight'):
                    modules["nn"].init.xavier_normal_(module.weight)
        else:
            raise ValueError(f"Unknown model: {cfg.model}")
        return model

    return _init()

def move_model_to_device(model, device, cfg, timers):
    @time_step("Model to Device", timers=timers, verbose=cfg.verbose.time)
    def _move():
        return model.to(device)
    return _move()

def setup_optimizer_and_loss(model, cfg, modules, timers):
    if cfg.verbose.build:
        print(f"[BUILD] Setting up optimizer and loss function...")
    nn, optim = modules["nn"], modules["optim"]
    @time_step("Init Optimizer & Loss", timers=timers, verbose=cfg.verbose.time)
    def _setup():
        loss = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=cfg.data.learning_rate, weight_decay=cfg.data.weight_decay)
        return loss, opt
    return _setup()

def ensure_save_directory(path, modules, cfg, timers):
    if cfg.verbose.build:
        print(f"[BUILD] Ensuring model save directory: {path}")
    os = modules["os"]
    @time_step("Ensure Save Directory", timers=timers, verbose=cfg.verbose.time)
    def _ensure():
        os.makedirs(path, exist_ok=True)
    _ensure()
    
def load_dataloaders(cfg, modules, timers):
    ZarrDataset = modules["ZarrDataset"]
    DataLoader = modules["torch"].utils.data.DataLoader
    random_split = modules["torch"].utils.data.random_split
    torch = modules["torch"]
    time_step = modules["time_step"]

    if cfg.verbose.build:
        print(f"[BUILD] Loading ZarrDataset...")

    @time_step("Dataloader Load", timers=timers, verbose=cfg.verbose.time)
    def _load():
        dataset = ZarrDataset(cfg.data.zarr_path)

        if cfg.verbose.build:
            print(f"[BUILD] Dataset contains {len(dataset.file_paths)} files.")
            print(f"[BUILD] Spatial dimensions: {dataset.data.shape[1:4]}")
            print(f"[BUILD] Subjects: {dataset.subject_ids}")
            print(f"[BUILD] Sessions: {dataset.session_ids}")
            print(f"[BUILD] Emotion categories: {dataset.emotions}")
            print(f"[BUILD] Total valid labeled timepoints: {len(dataset.valid_indices)}")

        train_size = int(cfg.train.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.data.seed)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle
        )

        return train_loader, val_loader

    return _load()