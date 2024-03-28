# train.py
import os
import datetime
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

def train(X_train,y_train,batch_size=32,num_epochs=10,num_workers=4,early_stopping_patience=10,device="cuda"):
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
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=num_epochs)


        # Create a directory to save the model weights
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float('inf')  # Initialize the best loss to infinity
    counter = 0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(num_epochs):
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
            scheduler.step()
            
            running_loss += loss.item() * images.size(0)

            progress_bar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        #Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at '{best_model_path}'")
        else:
            counter += 1
            if counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                early_stop_model_path = os.path.join(checkpoint_dir, f'early_stop_model_{timestamp}.pth')
                torch.save(model.state_dict(), early_stop_model_path)
                print(f"Early stop model saved at '{early_stop_model_path}'")
                break

        # Save a checkpoint every 5 epochs with timestamp
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

    print('Finished Training')

    # Save the final model with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    final_model_path = os.path.join(checkpoint_dir, f'final_model_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at '{final_model_path}'")
    
    
#if __name__ == "__main__":
 #   train(csv_file="data/train.csv",image_dir="data/images",batch_size=32,num_epochs=20,num_workers=4,early_stopping_patience=10,device="mps")