import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import hydra
import os
from models.resnet import ResNet, BasicBlock
import numpy as np
from utils.dataset import get_data_loaders
from omegaconf import DictConfig, OmegaConf
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
from math import ceil
import matplotlib.pyplot as plt


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


def visualize_and_save_gradcam(net, data_loader, num_images=10, output_dir="output/gradcam",batch_size=1,opacity=0.4):
    """
    Visualize and save Grad-CAM heatmaps as 3D volumes.
    Args:
        net: The trained network (e.g., your 3D ResNet).
        data_loader: DataLoader for the validation/test set.
        num_images: Number of images to process.
        output_dir: Directory to save the Grad-CAM outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    it = iter(data_loader)
    num_final_images = num_images % batch_size
    num_batches = ceil(num_images / batch_size)
    print('num_final_images',num_final_images)
    print('num_batches',num_batches)
    # Define the SpiceJet color map
    spicejet_colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # Blue -> Cyan -> Yellow -> Red
    spicejet_cmap = LinearSegmentedColormap.from_list('SpiceJet', spicejet_colors)
    for i in range(num_batches):
        img = next(it)["data_tensor"]
        img = img.cuda()  # Move to GPU if needed
        # Get the original shape of the image
        original_shape = img.shape  # (batch_size, channels, D, H, W)
        # Generate Grad-CAM heatmap for the 3D image
        img = img.float().to(device)  # Ensure data is float for model input
        if img.dim() == 4:
            img = img.unsqueeze(1)
        heatmap_3d = generate_gradcam_heatmap(net, img, original_shape=original_shape)
        if i + 1 == num_batches: # last batch
            internal_iter = num_final_images + 1
        else:
            internal_iter = batch_size
        for j in range(internal_iter):
            temp_heatmap = heatmap_3d[j]
            temp_heatmap = temp_heatmap.unsqueeze(0).unsqueeze(0)
            img_nii = nib.load(files[i*batch_size + j][0])
            img_array = img_nii.get_fdata()
            target_size = tuple(img_array.shape)
            heatmap = F.interpolate(temp_heatmap, size=target_size, mode='trilinear', align_corners=False)
            heatmap = heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()
            
            # Create a NIfTI image and save it
            output_filename = os.path.join(output_dir, f"gradcam.png")
            img_nii2 = nib.Nifti1Image(heatmap, img_nii.affine) 
            nib.save(img_nii2, output_filename)
            print(f"Saved {output_filename}")
            # Saving overlay to view in NiiVue:
            # Load original image
            original_img = img_nii
            original_data = original_img.get_fdata()
            # Normalize the original image to 0-255 range for RGB
            original_normalized = ((original_data - np.min(original_data)) * 255 /
                                (np.max(original_data) - np.min(original_data))).astype(np.uint8)
            # Create an RGB image from the original data
            rgb_image = np.zeros((*original_data.shape, 3), dtype=np.uint8)
            for i in range(3):
                rgb_image[..., i] = original_normalized
            # Load Grad-CAM mask
            gradcam_img = img_nii2
            gradcam_data = gradcam_img.get_fdata()
            # Normalize Grad-CAM values to 0-1 range
            gradcam_normalized = (gradcam_data - np.min(gradcam_data)) / (np.max(gradcam_data) - np.min(gradcam_data))
            # Map normalized Grad-CAM values to the SpiceJet colormap
            gradcam_colored = spicejet_cmap(gradcam_normalized)
            # Overlay Grad-CAM colors onto the original image with opacity
            for i in range(3):  # R, G, B channels
                rgb_image[..., i] = np.clip(
                    (1 - opacity) * rgb_image[..., i] + opacity * (gradcam_colored[..., i] * 255),
                    0, 255).astype(np.uint8)
            # Save the overlay as a new NIfTI file
            shape_3d = rgb_image.shape[:3]
            rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            ras_pos = rgb_image.copy().view(dtype=rgb_dtype).reshape(shape_3d)
            overlay_img = nib.Nifti1Image(ras_pos, original_img.affine)
            nib.save(overlay_img, os.path.join(output_dir, f"ogradcam.png"))
            print(f"Overlay saved as {os.path.join(output_dir, f"ogradcam.png")}")



def visualize_3d_heatmap(heatmap):
    """
    Visualize the 3d heatmap as a 3d volume.
    """
    data = heatmap.detach().numpy()

    x, y, z = np.indices(data.shape)
    values = data.flatten()
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111,projection='3d')
    norm = plt.Normalize(values.min(), values.max())
    yg = ax.scatter(x, y, z, c=values, marker='s', s=200, cmap="Greens_r")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig("output/gradcam/heatmap.png")



@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "output/models/s1tv1l6t_7.pth"

    train_dataloader, val_dataloader = get_data_loaders(cfg)
    model = ResNet(BasicBlock, [1, 1, 1, 1], in_channels=1, num_classes=22)
    print(model)
    # Initialize the new classification head with Xavier initialization
    def initialize_new_layers(model):
        for name, module in model.named_modules():
            if 'fc' in name:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    initialize_new_layers(model)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    
    visualize_and_save_gradcam(model, val_dataloader)

    # batch = next(iter(val_dataloader))
    # batch_data = batch["data_tensor"]
    # data = batch_data[0]

    # data = data.float().to(device)  # Ensure data is float for model input
    # data = data.unsqueeze(0).unsqueeze(0)
    # output = model(data)


    # heatmap = generate_gradcam_heatmap(model, data)

    # print(heatmap)
    # visualize_3d_heatmap(heatmap.squeeze())


if __name__ == "__main__":
    main()
