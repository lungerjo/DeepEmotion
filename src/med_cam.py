import os
import torch
import sys

PROJECT_ROOT = '/home/paperspace/DeepEmotion/src'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.dataset import get_data_loaders
from models.CNN import CNN, CNNNoPool
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from medcam import medcam
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def med_maps(cfg: DictConfig) -> None:
    # Get emotion mapping from config
    emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
    model = CNNNoPool(cfg=cfg, output_dim=6)
    model_path = '/home/paperspace/DeepEmotion/src/models/sub_01_NoPool.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    _, val_loader = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output/sub-01/No_Pool"
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
                if (batch["subject"][i] != 'sub-01'):
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
                    
                print(f"Saved visualization for batch {batch_idx}, {save_path} sample {i}")

class GradCAM3DNoPool:
    pass

class GradCAM3D:
    def __init__(self, model, target_layer, use_cuda=True):
        self.model = model
        self.cuda = use_cuda
        
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )
        
    def generate_cam(self, input_tensor, target_category=None):
        if len(input_tensor.shape) == 4:  
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, C, D, H, W)
            
        if target_category is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.argmax(dim=1).item()
                target_category = prediction
        else:
            prediction = target_category
        
        # Create target for pytorch_grad_cam
        targets = [ClassifierOutputTarget(target_category)]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        cam = grayscale_cam[0]  
        
        return cam, prediction
        
    def visualize_cam_3d(self, input_tensor, original_img, emotion_mapping, target_category=None, save_path=None):
        cam, prediction = self.generate_cam(input_tensor, target_category)

        print(cam.shape)

        if cam.shape != original_img.shape:
            print(f"Resizing CAM from {cam.shape} to {original_img.shape}")
            cam = resize(cam, 
                            output_shape=original_img.shape, 
                            order=1,  # Linear interpolation
                            preserve_range=True)
        
        print(cam.shape)
                
        # 3D data - visualize center slices from each axis
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        # Get the middle slices for each dimension
        d, h, w = original_img.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
        
        # Get middle slices of original image
        original_d = original_img[mid_d, :, :]
        original_h = original_img[:, mid_h, :]
        original_w = original_img[:, :, mid_w]
        
        # Get middle slices of CAM
        cam_d = cam[mid_d, :, :]
        cam_h = cam[:, mid_h, :]
        cam_w = cam[:, :, mid_w]
        
        # Create heatmap overlays
        axes[0, 0].imshow(original_d, cmap='gray')
        axes[0, 0].set_title('Original (Depth)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(original_d, cmap='gray')
        axes[0, 1].imshow(cam_d, cmap='jet', alpha=0.5)
        axes[0, 1].set_title('Grad-CAM (Depth)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(original_h, cmap='gray')
        axes[1, 0].set_title('Original (Height)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(original_h, cmap='gray')
        axes[1, 1].imshow(cam_h, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Grad-CAM (Height)')
        axes[1, 1].axis('off')
        
        axes[2, 0].imshow(original_w, cmap='gray')
        axes[2, 0].set_title('Original (Width)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(original_w, cmap='gray')
        axes[2, 1].imshow(cam_w, cmap='jet', alpha=0.5)
        axes[2, 1].set_title('Grad-CAM (Width)')
        axes[2, 1].axis('off')
        
        if target_category is not None and 'emotion_mapping' in globals() and target_category in emotion_mapping:
            emotion_name = emotion_mapping[target_category]
            plt.suptitle(f'Emotion: {emotion_name}', fontsize=16)
        else:
            plt.suptitle(f'Class: {prediction}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def visualize_average_cam_3d(self, input_tensor, original_img, emotion_mapping, target_category=None, save_path=None):
        """
        Visualize average Grad-CAM across all dimensions for 3D input
        
        Args:
            input_tensor: Input image tensor
            original_img: Original image (should be in range [0,1])
            target_category: Category to generate CAM for
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        cam, prediction = self.generate_cam(input_tensor, target_category)

        print(cam.shape)

        if cam.shape != original_img.shape:
            print(f"Resizing CAM from {cam.shape} to {original_img.shape}")
            cam = resize(cam, 
                            output_shape=original_img.shape, 
                            order=1,  # Linear interpolation
                            preserve_range=True)
        
        print(cam.shape)

        avg_d = np.mean(original_img, axis=0)  # Average across depth
        avg_h = np.mean(original_img, axis=1)  # Average across height
        avg_w = np.mean(original_img, axis=2)  # Average across width
        
        # Calculate average projections for CAM
        cam_avg_d = np.mean(cam, axis=0)  # Average across depth
        cam_avg_h = np.mean(cam, axis=1)  # Average across height
        cam_avg_w = np.mean(cam, axis=2)  # Average across width
        
        # Create figure with 3 rows and 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        # Row 1: Depth average projections
        axes[0, 0].imshow(avg_d, cmap='gray')
        axes[0, 0].set_title('Original (Depth Avg)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(avg_d, cmap='gray')
        axes[0, 1].imshow(cam_avg_d, cmap='jet', alpha=0.5)
        axes[0, 1].set_title('Grad-CAM (Depth Avg)')
        axes[0, 1].axis('off')
        
        # Row 2: Height average projections
        axes[1, 0].imshow(avg_h, cmap='gray')
        axes[1, 0].set_title('Original (Height Avg)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(avg_h, cmap='gray')
        axes[1, 1].imshow(cam_avg_h, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Grad-CAM (Height Avg)')
        axes[1, 1].axis('off')
        
        # Row 3: Width average projections
        axes[2, 0].imshow(avg_w, cmap='gray')
        axes[2, 0].set_title('Original (Width Avg)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(avg_w, cmap='gray')
        axes[2, 1].imshow(cam_avg_w, cmap='jet', alpha=0.5)
        axes[2, 1].set_title('Grad-CAM (Width Avg)')
        axes[2, 1].axis('off')
        
        # Add a super title with emotion name if available
        if target_category is not None and 'emotion_mapping' in globals() and target_category in emotion_mapping:
            emotion_name = emotion_mapping[target_category]
            plt.suptitle(f'Averaged Grad-CAM Projections\nEmotion: {emotion_name}', fontsize=16)
        else:
            plt.suptitle(f'Averaged Grad-CAM Projections\nClass: {prediction}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def other_maps(cfg: DictConfig) -> None:
    emotion_mapping = {v: k for k, v in cfg.data.emotion_idx.items()}
    
    model = CNN(cfg=cfg, output_dim=6)
    model_path = '/home/paperspace/DeepEmotion/src/models/sub_04.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    _, val_loader = get_data_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = "/home/paperspace/DeepEmotion/src/gradcam_output/other_library/conv1/"
    os.makedirs(output_dir, exist_ok=True)
    
    gradcam = GradCAM3D(model=model, target_layer=model.conv1, use_cuda=torch.cuda.is_available())
    
    for batch_idx, batch in enumerate(val_loader):
        print(f"Processing batch {batch_idx}")
        
        if batch_idx >= 5:  
            break
            
        for i in range(len(batch["data_tensor"])):
            if (batch["subject"][i] != 'sub-04'):
                continue
            print(f"Processing sample {i}")
            
            data = batch["data_tensor"][i:i+1].to(device)
            label = batch["label_tensor"][i:i+1].to(device)
            
            emotion_name = emotion_mapping.get(label.item(), f"Unknown ({label.item()})")
            
            original_img = data.squeeze(0).cpu().numpy()
            
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
            
            # save_path = os.path.join(output_dir, f"sample_{batch_idx}_{i}_emotion_{emotion_name}.png")
            # gradcam.visualize_cam_3d(data, original_img, target_category=label.item(), save_path=save_path, emotion_mapping=emotion_mapping)
            
            avg_save_path = os.path.join(output_dir, f"sample_{batch_idx}_{i}_emotion_{emotion_name}_avg.png")
            gradcam.visualize_average_cam_3d(data, original_img, target_category=label.item(), save_path=avg_save_path, emotion_mapping=emotion_mapping)
    
    print(f"Grad-CAM visualizations saved to {output_dir}")

@hydra.main(config_path="configs/", config_name="base", version_base="1.2")
def analyze_emotion_regions(cfg: DictConfig) -> None:
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
            avg_attention_map = np.mean(emotion_attention_maps[emotion], axis=0)

            fig, axes = plt.subplots(3, 2, figsize=(16, 24))
            
            # Get a representative brain scan for visualization
            sample_data = next(iter(val_loader))["data_tensor"][0].numpy()
            
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

            n_samples = len(emotion_attention_maps[emotion])
            avg_confidence = np.mean([pred['confidence'] for pred in emotion_correct_predictions[emotion]]) * 100
            
            title = f'Brain Regions Associated with {emotion}\n'
            title += f'Analysis based on {n_samples} correct predictions\n'
            title += f'Average confidence: {avg_confidence:.1f}%'
            
            plt.suptitle(title, fontsize=16, y=0.98)
            plt.tight_layout(pad=3.0)
            
            save_path = os.path.join(output_dir, f"{emotion}_brain_regions.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Completed analysis for {emotion} emotion")
            
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
    other_maps()
    # analyze_emotion_regions()
