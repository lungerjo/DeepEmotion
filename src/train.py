from utils.dataset import get_data_loader
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import LogisticRegressionModel


@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains a logistic regression model.
    """
    
    if cfg.verbose:
        print("Loading dataloader")
        
    dataloader = get_data_loader(cfg)

    # Set up model, loss function, and optimizer
    input_dim = 132 * 175 * 48  # Flattened input size
    output_dim = len(cfg.data.emotion_idx)  # Number of classes from emotion index
    model = LogisticRegressionModel(input_dim, output_dim)  # Multi-class logistic regression
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Training loop
    for epoch in range(cfg.train.epochs):  # Number of training epochs from config
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch, label in dataloader:
            # batch shape: [batch_size, 132, 175, 48]
            # label shape: [batch_size, num_classes] (one-hot encoded targets)

            flattened = batch.flatten(start_dim=1).float()  # Ensure input is in float32
            
            # Forward pass
            output = model(flattened)  # Shape [batch_size, output_dim]
            loss = criterion(output, label.argmax(dim=1))  # Convert one-hot labels to class indices
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Convert the output logits to predicted classes (take the argmax)
            _, predictions = torch.max(output, dim=1)
            
            # Convert one-hot encoded labels to class indices
            true_labels = label.argmax(dim=1)
            
            # Calculate number of correct predictions
            correct_predictions += (predictions == true_labels).sum().item()
            total_samples += label.size(0)
        
        # Calculate accuracy
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy*100:.2f}%")


    

if __name__ == "__main__":
    main()
