# train.py
import os
import datetime
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.data import plantDataset
from src.model import PlantModel
from src.utils import calculate_normalization_stats
from src.utils import init_weights

device = torch.device("cuda")
class R2Loss(torch.nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_true, y_pred):
        SS_res = torch.sum((y_true - y_pred)**2, axis=0)
        SS_tot = torch.sum((y_true - torch.mean(y_true, axis=0))**2, axis=0)
        r2_loss = SS_res / (SS_tot + 1e-6)
        return torch.mean(r2_loss)

class R2Metric(torch.nn.Module):
    def __init__(self):
        super(R2Metric, self).__init__()
        self.SS_res = torch.zeros(6).to(device)
        self.SS_tot = torch.zeros(6).to(device)
        self.num_samples = torch.zeros(6).to(device)

    def update_state(self, y_true, y_pred):
        SS_res = torch.sum((y_true - y_pred)**2, axis=0)
        SS_tot = torch.sum((y_true - torch.mean(y_true, axis=0))**2, axis=0)
        self.SS_res += SS_res
        self.SS_tot += SS_tot
        self.num_samples += y_true.shape[0]

    def forward(self):
        r2 = 1 - self.SS_res / (self.SS_tot + 1e-6)
        return torch.mean(r2)

    def reset_states(self):
        self.SS_res = torch.zeros(6).to(device)
        self.SS_tot = torch.zeros(6).to(device)
        self.num_samples = torch.zeros(6).to(device)
    
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
def get_data_split(XPath,yPath):
    with open(XPath, 'rb') as f:
            X = pickle.load(f)
    y = np.load(yPath)
    input_cols = [col for col in X.columns if not col.startswith('X') and col not in['id', 'file_path', 'jpeg_bytes']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_data = StandardScaler()
    X_train[input_cols] = scaler_data.fit_transform(X_train[input_cols])
    X_val[input_cols] = scaler_data.transform(X_val[input_cols])

    with open('data/X_Scaler.pkl', 'wb') as f:
        pickle.dump(scaler_data, f)
    return X_train, X_val, y_train, y_val
        
def train(X_train,y_train,batch_size=32,num_epochs=10,num_workers=4,early_stopping_patience=50,device="cuda",fresh_start=True):
    torch.manual_seed(420)
    device = torch.device(device)
    print("Using device:", device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a new instance of the plantDataset with the transformation pipeline
    #fullDataset = plantDataset(X_train=X_train, y_train=y_train, transform=transform)

    #train_size = int(0.8 * len(fullDataset))
    #test_size = len(fullDataset) - train_size
    #trainDataset, valDataset = torch.utils.data.random_split(fullDataset, [train_size, test_size])
    X_train, X_val, y_train, y_val = get_data_split(X_train,y_train)
    trainDataset = plantDataset(X_train, y_train, transform=transform)
    valDataset = plantDataset(X_val, y_val, transform=transform)
    
    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    valDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = PlantModel(num_ancillary_features=len(trainDataset.input_cols), num_output_features=trainDataset.y_train.shape[1])
    
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
    
    r2_metric = R2Metric().to(device)
    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        total_samples_train=0.0
        progress_bar = tqdm(trainDataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, input_data, target_data in progress_bar:
            images = images.to(device)
            input_data = input_data.float().to(device)
            target_data = target_data.float().to(device)
    
            optimizer.zero_grad()
            outputs = model(images, input_data)
            loss = criterion(outputs, target_data)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            r2_metric.update_state(target_data, outputs)
            total_samples_train += target_data.size(0)

            progress_bar.set_postfix(loss=loss.item())
        
        train_r2 = r2_metric()
        r2_metric.reset_states()

        model.eval()
        total_val_loss = 0
        total_samples_val = 0

        with torch.no_grad():
            for images, input_data,target_data in valDataloader:
                images = images.to(device)
                input_data = input_data.float().to(device)
                target_data = target_data.float().to(device)
                
                outputs = model(images, input_data)
                
                val_loss = criterion(target_data, outputs)
                total_val_loss += val_loss.item()

                # Obliczanie R^2 na zbiorze walidacyjnym
                r2_metric.update_state(target_data, outputs)
                total_samples_val += target_data.size(0)
            
            val_r2 = r2_metric()
            r2_metric.reset_states()
            
        epoch_loss = running_loss / len(trainDataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(trainDataloader):.4f}, Train R^2: {train_r2:.4f}, Val Loss: {total_val_loss/len(valDataloader):.4f}, Val R^2: {val_r2:.4f}")        
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

        # Save a checkpoint every 20 epochs with timestamp
        if (epoch + 1) % 20 == 0:
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