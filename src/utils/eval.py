import torch

def compute_masked_soft_accuracy(preds, targets, none_index, threshold=0.3):
    pred_class = preds.argmax(dim=1)
    true_class = targets.argmax(dim=1)
    mask = targets[:, none_index] < threshold
    correct = (pred_class[mask] == true_class[mask]).sum().item()
    total = mask.sum().item()
    return correct / (total + 1e-6)


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

def evaluate_soft_classification(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["data_tensor"].float().to(device)
            y = batch["label_tensor"].to(device)
            if x.dim() == 4:
                x = x.unsqueeze(1)
            logits = model(x)
            pred = logits.argmax(dim=1)
            target = y.argmax(dim=1)

            mask = (y[:, -1] <= 0.30)
            correct += (pred[mask] == target[mask]).sum().item()
            total += mask.sum().item()

    return correct / (total + 1e-6)

