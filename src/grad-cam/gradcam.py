import torch
import torch.nn.functional as F


def generate_gradcam_heatmap(net, img, original_shape=None):
    """Generate a 3D Grad-CAM heatmap for the input 3D image."""
    net.eval()
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    # Perform forward pass and gradient computation on GPU
    # img = img.cuda(non_blocking=True)
    pred = net(x=img, reg_hook=True)
    pred[:, pred.argmax(dim=1)].backward()

    # Retrieve gradients and activations
    gradients = net.get_activations_gradient().cpu()
    activations = net.get_activations(img).cpu()
    img = img.cpu()

    # Pool gradients
    pooled_gradients = gradients.mean(dim=(2, 3, 4), keepdim=True)

    # Weight activations
    weighted_activations = activations * pooled_gradients
    heatmap = weighted_activations.mean(dim=1)  # Average over channels

    # Apply ReLU and normalize
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-6  # Avoid divide-by-zero

    return heatmap
