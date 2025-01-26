import os
import torch
import sys

# Add the project root directory to the path
PROJECT_ROOT = '/home/paperspace/DeepEmotion/src'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.dataset import get_data_loaders
from models.CNN import CNN
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from medcam import medcam
from scipy.ndimage import zoom

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    model = CNN(cfg=cfg, output_dim=22)
    model_path = '/home/paperspace/DeepEmotion/src/output/models/tpkxxhdx_48.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    train_dataloader, _ = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample data shape
    sample_data = next(iter(train_dataloader))["data_tensor"][0:1]
    print(f"Sample data shape: {sample_data.shape}")
    
    # Calculate target size
    target_size = [s//2 for s in sample_data.shape[2:]]  # Halve each spatial dimension
    print(f"Target size: {target_size}")
    
    model = medcam.inject(
        model,
        output_dir=output_dir,
        backend="gcam",
        layer='conv2',
        save_maps=True
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"Processing batch {batch_idx}")
            
            for i in range(len(batch["data_tensor"])):
                print(f"Processing sample {i}")
                
                data = batch["data_tensor"][i:i+1].to(device)
                label = batch["label_tensor"][i:i+1].to(device)
                
                output = model(data)
                attention_map = model.get_attention_map()
                
                if attention_map is not None:
                    attention_map = attention_map.squeeze()
                    orig_data = batch["data_tensor"][i].cpu().numpy()
                    
                    print(f"Original data shape: {orig_data.shape}")
                    print(f"Attention map shape: {attention_map.shape}")
                    
                    # Get middle slices for each dimension
                    middle_x = orig_data.shape[0] // 2
                    middle_y = orig_data.shape[1] // 2
                    middle_z = orig_data.shape[2] // 2
                    
                    # Get attention map middle slices
                    att_middle_x = attention_map.shape[0] // 2
                    att_middle_y = attention_map.shape[1] // 2
                    att_middle_z = attention_map.shape[2] // 2
                    
                    # Create figure with three rows and two columns
                    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
                    
                    # Axial view (top-down)
                    axial_slice = orig_data[:, :, middle_z]
                    att_axial = attention_map[:, :, att_middle_z]
                    att_axial_resized = zoom(att_axial, 
                                           (orig_data.shape[0]/att_axial.shape[0], 
                                            orig_data.shape[1]/att_axial.shape[1]))
                    
                    axes[0, 0].imshow(axial_slice, cmap='gray')
                    axes[0, 0].set_title('Original - Axial View')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(axial_slice, cmap='gray')
                    im0 = axes[0, 1].imshow(att_axial_resized, cmap='jet', alpha=0.5)
                    axes[0, 1].set_title('GradCAM Overlay - Axial View')
                    plt.colorbar(im0, ax=axes[0, 1])
                    axes[0, 1].axis('off')
                    
                    # Coronal view (front-back)
                    coronal_slice = orig_data[:, middle_y, :].T
                    att_coronal = attention_map[:, att_middle_y, :].T
                    att_coronal_resized = zoom(att_coronal, 
                                             (orig_data.shape[2]/att_coronal.shape[0], 
                                              orig_data.shape[0]/att_coronal.shape[1]))
                    
                    axes[1, 0].imshow(coronal_slice, cmap='gray')
                    axes[1, 0].set_title('Original - Coronal View')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(coronal_slice, cmap='gray')
                    im1 = axes[1, 1].imshow(att_coronal_resized, cmap='jet', alpha=0.5)
                    axes[1, 1].set_title('GradCAM Overlay - Coronal View')
                    plt.colorbar(im1, ax=axes[1, 1])
                    axes[1, 1].axis('off')
                    
                    # Sagittal view (side)
                    sagittal_slice = orig_data[middle_x, :, :].T
                    att_sagittal = attention_map[att_middle_x, :, :].T
                    att_sagittal_resized = zoom(att_sagittal, 
                                              (orig_data.shape[2]/att_sagittal.shape[0], 
                                               orig_data.shape[1]/att_sagittal.shape[1]))
                    
                    axes[2, 0].imshow(sagittal_slice, cmap='gray')
                    axes[2, 0].set_title('Original - Sagittal View')
                    axes[2, 0].axis('off')
                    
                    axes[2, 1].imshow(sagittal_slice, cmap='gray')
                    im2 = axes[2, 1].imshow(att_sagittal_resized, cmap='jet', alpha=0.5)
                    axes[2, 1].set_title('GradCAM Overlay - Sagittal View')
                    plt.colorbar(im2, ax=axes[2, 1])
                    axes[2, 1].axis('off')
                    
                    plt.suptitle(f'GradCAM Analysis - Label: {label.item()}')
                    plt.tight_layout()
                    
                    # Save and cleanup
                    save_path = os.path.join(output_dir, f"batch_{batch_idx}_sample_{i}_all_views.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                print(f"Saved visualization for batch {batch_idx}, sample {i}")
            break
                
    

# @hydra.main(config_path="./configs", config_name="base", version_base="1.2")
# def show(cfg):
#     # Load Grad-CAM
#     attention_map = nib.load('/home/paperspace/DeepEmotion/src/gradcam_output/conv3/attention_map_0_0_0.nii.gz')
#     attention_data = attention_map.get_fdata()
    
#     train_dataloader, _ = get_data_loaders(cfg)
#     # Load original image
#     first_batch = next(iter(train_dataloader))
#     original = first_batch["data_tensor"][0,0].cpu().numpy()
    
#     # Resize attention map to match original
#     from scipy.ndimage import zoom
#     zoom_factors = (original.shape[0]/attention_data.shape[0], 
#                    original.shape[1]/attention_data.shape[1],
#                    original.shape[2]/attention_data.shape[2])
#     attention_resized = zoom(attention_data, zoom_factors)
    
#     # Overlay with proper normalization
#     plt.figure(figsize=(12,6))
#     plt.subplot(1,2,1)
#     plt.imshow(original[..., original.shape[-1]//2], cmap='gray')
#     plt.title('Original')
    
#     plt.subplot(1,2,2)
#     plt.imshow(original[..., original.shape[-1]//2], cmap='gray')
#     plt.imshow(attention_resized[..., attention_resized.shape[-1]//2], 
#               alpha=0.4, cmap='jet')
#     plt.title('Grad-CAM Overlay')
    
#     plt.savefig('/home/paperspace/DeepEmotion/src/gradcam_output/conv3/overlay.png')
#     plt.close()

if __name__ == "__main__":
   main()
