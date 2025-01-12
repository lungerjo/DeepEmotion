import torch
import yaml

import torch.nn as nn
import torch.nn.functional as F

class VGG16Network(nn.Module):
    def __init__(self, output_dim=1):
        super(VGG16Network, self).__init__()
        # Define the VGG network architecture
        self.features = nn.Sequential(
            self._convrelublock(10, 64, 2),
            self._convrelublock(64, 128, 2),
            self._convrelublock(128, 256, 3),
            self._convrelublock(256, 512, 3),
            self._convrelublock(512, 512, 3)
        )
            # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 5 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),)
    
    def _convrelublock(self, in_channels, out_channels, layers):
        block = []
        for _ in range(layers):
            block.append(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1))
            block.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        block.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*block)
      

    def forward(self, x):
        x = self.features(x)
        print(f"Size after features: {x.size()}")  # Print size after features
        x = torch.flatten(x, 0)
        print(f"Size after flatten: {x.size()}")
        x = self.classifier(x)
        return x
        

# Example usage
if __name__ == "__main__":
    input_dim = (10, 132, 175, 48)
    with open('configs/base.yaml', 'r') as file:
        config = yaml.safe_load(file)
    output_dim = len(config['data']['emotion_idx'])
    model = VGG16Network(output_dim=output_dim)
    print(model)
    dummy_input = torch.randn(input_dim)
    output = model(dummy_input)
    print(f"Output size: {output.size()}")