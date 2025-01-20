from resnet3d import ResNet3D
import torch
import os
from solids_dataset import SolidDataset
from gradcam import generate_gradcam_heatmap
from utils import visualize_3d_heatmap


model_path = os.path.join("logs/2025-01-19_11-44-07/resnet3d_epoch_1")
model = ResNet3D(base_channels = 4)
model.load_state_dict(torch.load(model_path))
model.eval()

dataset = SolidDataset('./data_solid', './labels_solid/label_train_set.npy')
input = dataset[200][0].unsqueeze(0)
output = model(input)
predicted = (output.squeeze() > 0.5).float().item()

heatmap = generate_gradcam_heatmap(model, input)
print(heatmap)
visualize_3d_heatmap(heatmap.squeeze())
