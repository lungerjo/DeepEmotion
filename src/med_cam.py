import os
import torch
import sys

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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nilearn
from nilearn import plotting
import nibabel as nib


def create_nifti_image(data):
    """Convert numpy array to NIfTI image with identity affine."""
    affine = np.eye(4)  # Identity affine transformation
    return nib.Nifti1Image(data, affine)

def process_attention_map(attention_map, threshold=0.4):
    """Process attention map to remove background noise."""
    # Normalize to [-1, 1] range
    norm_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 2 - 1
    
    # Apply threshold
    norm_map[np.abs(norm_map) < threshold] = 0
    
    return norm_map

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
    model = CNN(cfg=cfg, output_dim=22)
    model_path = '/home/paperspace/DeepEmotion/src/output/models/tpkxxhdx_48.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    train_dataloader, _ = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output_nilearn"
    os.makedirs(output_dir, exist_ok=True)
    
    model = medcam.inject(
        model,
        output_dir=output_dir,
        backend="gcam",
        layer='conv2',
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            for i in range(len(batch["data_tensor"])):
                print(f"Processing sample {i}")
                
                data = batch["data_tensor"][i:i+1].to(device)
                label = batch["label_tensor"][i:i+1].to(device)
                subject = batch['subject'][i]
                time_offset = batch['time_offset'][i]
                
                emotion_name = emotion_mapping.get(label.item(), f"Unknown ({label.item()})")
                
                output = model(data)
                attention_map = model.get_attention_map()

                if attention_map is not None:
                    attention_map = attention_map.squeeze()
                    orig_data = batch["data_tensor"][i].cpu().numpy()
                    
                    # Resize attention map to match original volume size
                    resized_attention = zoom(attention_map, 
                                          (orig_data.shape[0]/attention_map.shape[0],
                                           orig_data.shape[1]/attention_map.shape[1],
                                           orig_data.shape[2]/attention_map.shape[2]))
                    
                    # Process attention map
                    processed_attention = process_attention_map(resized_attention)
                    
                    # Create NIfTI images
                    brain_img = create_nifti_image(orig_data)
                    attention_img = create_nifti_image(processed_attention)
                    
                    # Create figure with multiple views
                    fig = plt.figure(figsize=(20, 10))
                    
                    # Plot original brain
                    ax1 = plt.subplot(121)
                    display = plotting.plot_anat(brain_img, 
                                               display_mode='ortho',
                                               cut_coords=(0, 0, 0),
                                               title='Original Brain Scan',
                                               axes=ax1,
                                               annotate=True)
                    
                    # Plot GradCAM overlay
                    ax2 = plt.subplot(122)
                    display = plotting.plot_stat_map(attention_img,
                                                   bg_img=brain_img,
                                                   display_mode='ortho',
                                                   cut_coords=(0, 0, 0),
                                                   colorbar=True,
                                                   cmap='RdBu_r',  # Red-Blue diverging colormap
                                                   title='GradCAM Attention',
                                                   axes=ax2,
                                                   threshold=0.4,  # Only show significant activations
                                                   annotate=True,
                                                   black_bg=True)  # Black background
                    
                    # Add overall title with metadata
                    plt.suptitle(f'3D Brain Analysis\nEmotion: {emotion_name}\nSubject: {subject}\nTime Offset: {time_offset}s',
                               fontsize=16, y=1.1)
                    
                    # Save visualization
                    save_path = os.path.join(output_dir, 
                                           f"nilearn_sub{subject}_time{time_offset}_emotion{emotion_name}.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    print(f"Saved visualization for subject {subject}, emotion {emotion_name}")
                
            break

if __name__ == "__main__":
    main()

# @hydra.main(config_path="configs/", config_name="base", version_base="1.2")
# def main(cfg: DictConfig) -> None:
#     # Get emotion mapping from config
#     emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
#     model = CNN(cfg=cfg, output_dim=22)
#     model_path = '/home/paperspace/DeepEmotion/src/output/models/tpkxxhdx_48.pth'
#     model.load_state_dict(torch.load(model_path, weights_only=True))
#     model.eval()

#     train_dataloader, _ = get_data_loaders(cfg)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output"
#     os.makedirs(output_dir, exist_ok=True)
    
#     sample_data = next(iter(train_dataloader))["data_tensor"][0:1]
#     print(f"Sample data shape: {sample_data.shape}")
    
#     target_size = [s//2 for s in sample_data.shape[2:]]
#     print(f"Target size: {target_size}")
    
#     model = medcam.inject(
#         model,
#         output_dir=output_dir,
#         backend="gcam",
#         layer='conv2',
#     )

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(train_dataloader):
#             print(f"Processing batch {batch_idx}")
            
#             for i in range(len(batch["data_tensor"])):
#                 print(f"Processing sample {i}")
                
#                 data = batch["data_tensor"][i:i+1].to(device)
#                 label = batch["label_tensor"][i:i+1].to(device)
                
#                 # Get emotion name from label
#                 emotion_name = emotion_mapping.get(label.item(), f"Unknown ({label.item()})")
                
#                 output = model(data)
#                 attention_map = model.get_attention_map()
                
#                 if attention_map is not None:
#                     attention_map = attention_map.squeeze()
#                     orig_data = batch["data_tensor"][i].cpu().numpy()
                    
#                     print(f"Original data shape: {orig_data.shape}")
#                     print(f"Attention map shape: {attention_map.shape}")
                    
#                     middle_x = orig_data.shape[0] // 2
#                     middle_y = orig_data.shape[1] // 2
#                     middle_z = orig_data.shape[2] // 2
                    
#                     att_middle_x = attention_map.shape[0] // 2
#                     att_middle_y = attention_map.shape[1] // 2
#                     att_middle_z = attention_map.shape[2] // 2
                    
#                     fig, axes = plt.subplots(3, 2, figsize=(16, 24))
                    
#                     # Axial view
#                     axial_slice = orig_data[:, :, middle_z]
#                     att_axial = attention_map[:, :, att_middle_z]
#                     att_axial_resized = zoom(att_axial, 
#                                            (orig_data.shape[0]/att_axial.shape[0], 
#                                             orig_data.shape[1]/att_axial.shape[1]))
                    
#                     axes[0, 0].imshow(axial_slice, cmap='gray')
#                     axes[0, 0].set_title('Original - Axial View')
#                     axes[0, 0].axis('off')
                    
#                     axes[0, 1].imshow(axial_slice, cmap='gray')
#                     im0 = axes[0, 1].imshow(att_axial_resized, cmap='jet', alpha=0.5)
#                     axes[0, 1].set_title('GradCAM Overlay - Axial View')
#                     plt.colorbar(im0, ax=axes[0, 1])
#                     axes[0, 1].axis('off')
                    
#                     # Coronal view
#                     coronal_slice = orig_data[:, middle_y, :].T
#                     att_coronal = attention_map[:, att_middle_y, :].T
#                     att_coronal_resized = zoom(att_coronal, 
#                                              (orig_data.shape[2]/att_coronal.shape[0], 
#                                               orig_data.shape[0]/att_coronal.shape[1]))
                    
#                     axes[1, 0].imshow(coronal_slice, cmap='gray')
#                     axes[1, 0].set_title('Original - Coronal View')
#                     axes[1, 0].axis('off')
                    
#                     axes[1, 1].imshow(coronal_slice, cmap='gray')
#                     im1 = axes[1, 1].imshow(att_coronal_resized, cmap='jet', alpha=0.5)
#                     axes[1, 1].set_title('GradCAM Overlay - Coronal View')
#                     plt.colorbar(im1, ax=axes[1, 1])
#                     axes[1, 1].axis('off')
                    
#                     # Sagittal view
#                     sagittal_slice = orig_data[middle_x, :, :].T
#                     att_sagittal = attention_map[att_middle_x, :, :].T
#                     att_sagittal_resized = zoom(att_sagittal, 
#                                               (orig_data.shape[2]/att_sagittal.shape[0], 
#                                                orig_data.shape[1]/att_sagittal.shape[1]))
                    
#                     axes[2, 0].imshow(sagittal_slice, cmap='gray')
#                     axes[2, 0].set_title('Original - Sagittal View')
#                     axes[2, 0].axis('off')
                    
#                     axes[2, 1].imshow(sagittal_slice, cmap='gray')
#                     im2 = axes[2, 1].imshow(att_sagittal_resized, cmap='jet', alpha=0.5)
#                     axes[2, 1].set_title('GradCAM Overlay - Sagittal View')
#                     plt.colorbar(im2, ax=axes[2, 1])
#                     axes[2, 1].axis('off')
                    
#                     # Update title to show emotion name, subject, and time offset
#                     subject = batch['subject'][i]
#                     time_offset = batch['time_offset'][i]
#                     plt.suptitle(f'GradCAM Analysis\nEmotion: {emotion_name}\nSubject: {subject}\nTime Offset: {time_offset}s', 
#                                 fontsize=16, y=0.98)
#                     plt.tight_layout(pad=3.0)
                    
#                     # Save and cleanup
#                     save_path = os.path.join(output_dir, f"batch_{batch_idx}_sample_{i}_all_views.png")
#                     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#                     plt.close()
                    
#                 print(f"Saved visualization for batch {batch_idx}, sample {i}")
#             break
