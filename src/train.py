# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.data import plantDataset
from src.model import ResNetFiLM
from src.utils import calculate_normalization_stats

def train(csv_file,image_dir,batch_size=32,num_epochs=10,num_workers=4):
    # Create an instance of the plantDataset (without transforms)
    dataset = plantDataset(csv_file=csv_file, image_dir=image_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a new instance of the plantDataset with the transformation pipeline
    dataset = plantDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create an instance of the model
    model = ResNetFiLM(num_input_features=len(dataset.input_cols), num_output_features=len(dataset.target_cols))

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    # Move the model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

        # Create a directory to save the model weights
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float('inf')  # Initialize the best loss to infinity

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, input_data, target_data in progress_bar:
            images = images.to(device)
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            
            # Normalize input data and ancillary data
            input_data_mean = input_data.mean(dim=0)
            input_data_std = input_data.std(dim=0)
            input_data = (input_data - input_data_mean.to(device)) / input_data_std.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, input_data)
            loss = criterion(outputs, target_data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # Save the model weights if the current loss is better than the best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model.pth'))
        
        # Save the model weights at regular intervals (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
    print('Finished Training')
    # Save the final model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))