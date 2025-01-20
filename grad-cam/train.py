from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from resnet3d import ResNet3D
from solids_dataset import SolidDataset


# initialize datesets
train_data_dir = './data_solid'
train_label_dir = './labels_solid/label_train_set.npy'
test_data_dir = './test_solid'
test_label_dir = './labels_solid/label_test_set.npy'
train_dataset = SolidDataset(train_data_dir, train_label_dir)
test_dataset = SolidDataset(test_data_dir, test_label_dir)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = ResNet3D(base_channels=4)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training setup
num_epochs = 10
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join("logs", current_time)
os.makedirs(folder_path, exist_ok=True)

for epoch in range(num_epochs):
    # training segment
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs, labels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_acc)
    
    # evaluation segment
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            running_test_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    
    # calculate test accuracy and loss
    avg_test_loss = running_test_loss / len(test_loader)
    test_acc = 100 * correct_test / total_test
    test_loss_history.append(avg_test_loss)
    test_acc_history.append(test_acc)
    
    # report and save
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(folder_path, f"resnet3d_epoch_{epoch+1}"))
