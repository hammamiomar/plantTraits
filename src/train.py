# train.py
import os
import datetime
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.data import plantDataset
from src.model import PlantModel
from src.utils import calculate_normalization_stats
from src.utils import init_weights

class R2Loss(torch.nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_true, y_pred):
        SS_res = torch.sum((y_true - y_pred)**2, axis=0)
        SS_tot = torch.sum((y_true - torch.mean(y_true, axis=0))**2, axis=0)
        r2_loss = SS_res / (SS_tot + 1e-6)
        return torch.mean(r2_loss)
    
def find_latest_checkpoint(checkpoint_dir):
    # Define the regex pattern for matching filenames and extracting epoch numbers
    pattern = re.compile(r"checkpoint_epoch_(\d+).pth")

    # Initialize variables to keep track of the highest epoch and corresponding file
    highest_epoch = -1
    latest_checkpoint_file = None

    # Iterate over all files in the checkpoint directory
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            # Extract the epoch number from the filename
            epoch_num = int(match.group(1))
            if epoch_num > highest_epoch:
                highest_epoch = epoch_num
                latest_checkpoint_file = filename

    return latest_checkpoint_file

def train(X_train,y_train,batch_size=32,num_epochs=10,num_workers=4,early_stopping_patience=50,device="cuda",fresh_start=True):
    torch.manual_seed(420)
    # Create an instance of the plantDataset (without transforms)
    #device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    #ddevice="mps"
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a new instance of the plantDataset with the transformation pipeline
    dataset = plantDataset(X_train=X_train, y_train=y_train, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create an instance of the model
    
    model = PlantModel(num_ancillary_features=len(dataset.input_cols), num_output_features=dataset.y_train.shape[1])
    
    #model.apply(init_weights)
    model.to(device)

    # Define loss function and optimizer
    criterion = R2Loss()  # Mean Average Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=num_epochs)

    start_epoch = 0
    best_loss = float('inf')  # Initialize the best loss to infinity
    #counter = 0
    
        # Create a directory to save the model weights
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latestCheckpoint=find_latest_checkpoint(checkpoint_dir)
    if latestCheckpoint and fresh_start==False:
        checkpoint_path = os.path.join(checkpoint_dir, latestCheckpoint)
        checkpoint = torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint from '{checkpoint_path}' at epoch {start_epoch}")
    
    
    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, input_data, target_data in progress_bar:
            images = images.to(device)
            input_data = input_data.float().to(device)
            target_data = target_data.float().to(device)
            
            # Normalize input data and ancillary data
            #input_data_mean = input_data.mean(dim=0)
            #input_data_std = input_data.std(dim=0)
            #input_data = (input_data - input_data_mean.to(device)) / input_data_std.to(device)
            #target_data = (target_data - torch.from_numpy(target_mean).float().to(device)) / torch.from_numpy(target_std).float().to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, input_data)
            loss = criterion(outputs, target_data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            running_loss += loss.item() * images.size(0)

            progress_bar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        #Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
            best_model_path = os.path.join(checkpoint_dir, f'best_model.pth')
            torch.save({
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_loss':best_loss}, best_model_path)
            print(f"Best model saved at '{best_model_path}'")
        else:
            counter += 1
            if counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop_model_path = os.path.join(checkpoint_dir, f'early_stop_model.pth')
                torch.save({
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_loss':best_loss}, early_stop_model_path)
                print(f"Early stop model saved at '{early_stop_model_path}'")
                break

        # Save a checkpoint every 5 epochs with timestamp
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_loss':best_loss}, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

    print('Finished Training')

    # Save the final model with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    final_model_path = os.path.join(checkpoint_dir, f'final_model_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at '{final_model_path}'")
    
    
#if __name__ == "__main__":
 #   train(csv_file="data/train.csv",image_dir="data/images",batch_size=32,num_epochs=20,num_workers=4,early_stopping_patience=10,device="mps")