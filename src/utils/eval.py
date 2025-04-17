import torch

def evaluate_classification(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch["data_tensor"].to(device)
            labels = batch["label_tensor"].to(device)
            if data.dim() == 4:
                data = data.unsqueeze(1)
            preds = model(data).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def evaluate_soft_classification(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch["data_tensor"].to(device)
            labels = batch["label_tensor"].to(device)
            if data.dim() == 4:
                data = data.unsqueeze(1)
            preds = model(data)
            pred_class = preds.argmax(dim=1)
            true_class = labels.argmax(dim=1)
            correct += (pred_class == true_class).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0
