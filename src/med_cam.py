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
import torch.nn.functional as F
import json

# import nilearn
# from nilearn import plotting
# import nibabel as nib

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Get emotion mapping from config
    emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
    model = CNN(cfg=cfg, output_dim=6)
    model_path = '/home/paperspace/DeepEmotion/src/models/sub_04.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    _, val_loader = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output/sub-04"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_data = next(iter(val_loader))["data_tensor"][0:1]
    print(f"Sample data shape: {sample_data.shape}")
    
    target_size = [s//2 for s in sample_data.shape[2:]]
    print(f"Target size: {target_size}")
    
    model = medcam.inject(
        model,
        output_dir=output_dir,
        backend="gcam",
        layer='conv3',
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"Processing batch {batch_idx}")
            
            for i in range(len(batch["data_tensor"])):
                if (batch["subject"][i] != 'sub-04'):
                    continue
                print(f"Processing sample {i}")
                
                data = batch["data_tensor"][i:i+1].to(device)
                label = batch["label_tensor"][i:i+1].to(device)
                
                # Get emotion name from label
                emotion_name = emotion_mapping.get(label.item(), f"Unknown ({label.item()})")
                
                output = model(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, dim=1)
                pred_emotion = emotion_mapping.get(predicted.item(), f"Unknown ({predicted.item()})")

                attention_map = model.get_attention_map()
                
                if attention_map is not None:
                    attention_map = attention_map.squeeze()
                    orig_data = batch["data_tensor"][i].cpu().numpy()
                    
                    print(f"Original data shape: {orig_data.shape}")
                    print(f"Attention map shape: {attention_map.shape}")
                    
                    # middle_x = orig_data.shape[0] // 2
                    # middle_y = orig_data.shape[1] // 2
                    # middle_z = orig_data.shape[2] // 2
                    
                    # att_middle_x = attention_map.shape[0] // 2
                    # att_middle_y = attention_map.shape[1] // 2
                    # att_middle_z = attention_map.shape[2] // 2
                    
                    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
                    
                    axial_slice = orig_data.mean(axis=2)  
                    att_axial = attention_map.mean(axis=2)  # Average across the same depth dimension
                    att_axial_resized = F.interpolate(
                        torch.tensor(att_axial).unsqueeze(0).unsqueeze(0), 
                        size=(orig_data.shape[0], orig_data.shape[1]), 
                        mode='bicubic', 
                        align_corners=False
                    ).squeeze().numpy()
                    
                    axes[0, 0].imshow(axial_slice, cmap='gray')
                    axes[0, 0].set_title('Original - Axial View')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(axial_slice, cmap='gray')
                    im0 = axes[0, 1].imshow(att_axial_resized, cmap='jet', alpha=0.5)
                    axes[0, 1].set_title('GradCAM Overlay - Axial View')
                    plt.colorbar(im0, ax=axes[0, 1])
                    axes[0, 1].axis('off')
                    
                    # Coronal view
                    coronal_slice = np.flipud(orig_data.mean(axis=1).T).copy()
                    att_coronal = np.flipud(attention_map.mean(axis=1).T).copy()  # Average across the same height dimension
                    att_coronal_resized = F.interpolate(
                        torch.tensor(att_coronal).unsqueeze(0).unsqueeze(0), 
                        size=(orig_data.shape[2], orig_data.shape[0]), 
                        mode='bicubic', 
                        align_corners=False
                    ).squeeze().numpy()
                    
                    axes[1, 0].imshow(coronal_slice, cmap='gray')
                    axes[1, 0].set_title('Original - Coronal View')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(coronal_slice, cmap='gray')
                    im1 = axes[1, 1].imshow(att_coronal_resized, cmap='jet', alpha=0.5)
                    axes[1, 1].set_title('GradCAM Overlay - Coronal View')
                    plt.colorbar(im1, ax=axes[1, 1])
                    axes[1, 1].axis('off')
                    
                    # Sagittal view
                    sagittal_slice = np.flipud(orig_data.mean(axis=0).T).copy()  
                    att_sagittal = np.flipud(attention_map.mean(axis=0).T).copy() # Average across depth
                    att_sagittal_resized = F.interpolate(
                        torch.tensor(att_sagittal).unsqueeze(0).unsqueeze(0), 
                        size=(orig_data.shape[2], orig_data.shape[1]), 
                        mode='bicubic', 
                        align_corners=False
                    ).squeeze().numpy()
                    
                    axes[2, 0].imshow(sagittal_slice, cmap='gray')
                    axes[2, 0].set_title('Original - Sagittal View')
                    axes[2, 0].axis('off')
                    
                    axes[2, 1].imshow(sagittal_slice, cmap='gray')
                    im2 = axes[2, 1].imshow(att_sagittal_resized, cmap='jet', alpha=0.5)
                    axes[2, 1].set_title('GradCAM Overlay - Sagittal View')
                    plt.colorbar(im2, ax=axes[2, 1])
                    axes[2, 1].axis('off')

                    subject = batch['subject'][i]
                    time_offset = batch['time_offset'][i]
                    title = f'GradCAM Analysis\n'
                    title += f'True Emotion: {emotion_name}\n'
                    title += f'Predicted Emotion: {pred_emotion} ({probabilities[0][predicted].item()*100:.1f}%)\n'
                    title += f'Subject: {subject}\nTime Offset: {time_offset}s'
                    
                    plt.suptitle(title, fontsize=16, y=0.98)
                    plt.tight_layout(pad=3.0)
                    
                    # Save and cleanup
                    save_path = os.path.join(output_dir, f"batch_{batch_idx}_sample_{i}_all_views.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                print(f"Saved visualization for batch {batch_idx}, sample {i}")

# if __name__ == "__main__":
#     main()

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def analyze_emotion_regions(cfg: DictConfig) -> None:
    # Get emotion mapping from config
    emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
    model = CNN(cfg=cfg, output_dim=6)
    model_path = '/home/paperspace/DeepEmotion/src/models/sub_04.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    _, val_loader = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output/emotion_regions_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionaries to store aggregated attention maps for each emotion
    emotion_attention_maps = {emotion: [] for emotion in emotion_mapping.values()}
    emotion_correct_predictions = {emotion: [] for emotion in emotion_mapping.values()}
    
    model = medcam.inject(
        model,
        output_dir=output_dir,
        backend="gcam",
        layer='conv3',
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"Processing batch {batch_idx}")
            
            for i in range(len(batch["data_tensor"])):
                data = batch["data_tensor"][i:i+1].to(device)
                label = batch["label_tensor"][i:i+1].to(device)
                
                # Get emotion name from label
                true_emotion = emotion_mapping.get(label.item(), f"Unknown ({label.item()})")
                
                output = model(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, dim=1)
                pred_emotion = emotion_mapping.get(predicted.item(), f"Unknown ({predicted.item()})")

                # Only store attention maps for correct predictions
                if true_emotion == pred_emotion:
                    attention_map = model.get_attention_map()
                    if attention_map is not None:
                        attention_map = attention_map.squeeze()
                        orig_data = batch["data_tensor"][i].cpu().numpy()
                        emotion_attention_maps[true_emotion].append(attention_map)
                        emotion_correct_predictions[true_emotion].append({
                            'subject': batch['subject'][i],
                            'time_offset': batch['time_offset'][i],
                            'confidence': probabilities[0][predicted].item()
                        })

    # Create averaged attention maps and visualizations for each emotion
    for emotion in emotion_mapping.values():
        if len(emotion_attention_maps[emotion]) > 0:
            # Average attention maps for this emotion
            avg_attention_map = np.mean(emotion_attention_maps[emotion], axis=0)
            
            # Create visualization
            fig, axes = plt.subplots(3, 2, figsize=(16, 24))
            
            # Get a representative brain scan for visualization
            sample_data = next(iter(val_loader))["data_tensor"][0].numpy()
            
            # Axial view
            axial_slice = sample_data.mean(axis=2)
            att_axial = avg_attention_map.mean(axis=2)
            att_axial_resized = F.interpolate(
                torch.tensor(att_axial).unsqueeze(0).unsqueeze(0),
                size=(sample_data.shape[0], sample_data.shape[1]),
                mode='bicubic',
                align_corners=False
            ).squeeze().numpy()
            
            axes[0, 0].imshow(axial_slice, cmap='gray')
            axes[0, 0].set_title('Brain - Axial View')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(axial_slice, cmap='gray')
            im0 = axes[0, 1].imshow(att_axial_resized, cmap='jet', alpha=0.5)
            axes[0, 1].set_title(f'{emotion} Attention - Axial View')
            plt.colorbar(im0, ax=axes[0, 1])
            axes[0, 1].axis('off')
            
            # Coronal view
            coronal_slice = np.flipud(sample_data.mean(axis=1).T).copy()
            att_coronal = np.flipud(avg_attention_map.mean(axis=1).T).copy()
            att_coronal_resized = F.interpolate(
                torch.tensor(att_coronal).unsqueeze(0).unsqueeze(0),
                size=(sample_data.shape[2], sample_data.shape[0]),
                mode='bicubic',
                align_corners=False
            ).squeeze().numpy()
            
            axes[1, 0].imshow(coronal_slice, cmap='gray')
            axes[1, 0].set_title('Brain - Coronal View')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(coronal_slice, cmap='gray')
            im1 = axes[1, 1].imshow(att_coronal_resized, cmap='jet', alpha=0.5)
            axes[1, 1].set_title(f'{emotion} Attention - Coronal View')
            plt.colorbar(im1, ax=axes[1, 1])
            axes[1, 1].axis('off')
            
            # Sagittal view
            sagittal_slice = np.flipud(sample_data.mean(axis=0).T).copy()
            att_sagittal = np.flipud(avg_attention_map.mean(axis=0).T).copy()
            att_sagittal_resized = F.interpolate(
                torch.tensor(att_sagittal).unsqueeze(0).unsqueeze(0),
                size=(sample_data.shape[2], sample_data.shape[1]),
                mode='bicubic',
                align_corners=False
            ).squeeze().numpy()
            
            axes[2, 0].imshow(sagittal_slice, cmap='gray')
            axes[2, 0].set_title('Brain - Sagittal View')
            axes[2, 0].axis('off')
            
            axes[2, 1].imshow(sagittal_slice, cmap='gray')
            im2 = axes[2, 1].imshow(att_sagittal_resized, cmap='jet', alpha=0.5)
            axes[2, 1].set_title(f'{emotion} Attention - Sagittal View')
            plt.colorbar(im2, ax=axes[2, 1])
            axes[2, 1].axis('off')

            # Add summary statistics
            n_samples = len(emotion_attention_maps[emotion])
            avg_confidence = np.mean([pred['confidence'] for pred in emotion_correct_predictions[emotion]]) * 100
            
            title = f'Brain Regions Associated with {emotion}\n'
            title += f'Analysis based on {n_samples} correct predictions\n'
            title += f'Average confidence: {avg_confidence:.1f}%'
            
            plt.suptitle(title, fontsize=16, y=0.98)
            plt.tight_layout(pad=3.0)
            
            # Save visualization
            save_path = os.path.join(output_dir, f"{emotion}_brain_regions.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Completed analysis for {emotion} emotion")
            
    # Save summary statistics
    summary = {
        'emotion_samples': {emotion: len(maps) for emotion, maps in emotion_attention_maps.items()},
        'average_confidence': {
            emotion: np.mean([pred['confidence'] for pred in preds]) * 100 
            for emotion, preds in emotion_correct_predictions.items() 
            if preds
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    analyze_emotion_regions()
