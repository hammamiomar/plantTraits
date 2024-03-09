import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


class plantDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Get the column names for input features and target variables
        self.input_cols = [col for col in self.data.columns if not col.startswith('X') and col != 'id']
        self.target_cols = [col for col in self.data.columns if col.startswith('X') and col.endswith('_mean')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.data.iloc[idx, 0]}.jpeg")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Get the input features and target variables separately
        input_data = self.data.iloc[idx][self.input_cols].values.astype(np.float32)
        target_data = self.data.iloc[idx][self.target_cols].values.astype(np.float32)

        return image, input_data, target_data, img_path

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()# Resize the images to a smaller size
])
