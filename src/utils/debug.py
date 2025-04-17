import torch.nn.functional as F

def print_soft_label(outputs, labels, batch_idx, epoch):
    for i in range(min(3, labels.size(0))):
        pred_probs = F.softmax(outputs[i], dim=0).detach().cpu().numpy()
        true_class = labels[i].argmax().item()
        pred_class = pred_probs.argmax()
        print(f"[DEBUG][Epoch {epoch+1} | Batch {batch_idx}]")
        print(f"True soft label: {labels[i].cpu().numpy().round(2)} (argmax: {true_class})")
        print(f"Predicted logits: {outputs[i]} (argmax: {pred_class})")
        print(f"Predicted probs: {pred_probs.round(2)} (argmax: {pred_class})")
