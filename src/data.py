import os
import pandas as pd
from torch.utils.data import Dataset
import imageio.v3 as  imageio
import numpy as np
import pickle
from io import BytesIO
from PIL import Image


class plantDataset(Dataset):
    def __init__(self, X_train, y_train, transform=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_jpegs = self.X_train['jpeg_bytes'].values
        self.transform = transform

        # Get the column names for input features and target variables
        self.input_cols = [col for col in self.X_train.columns if not col.startswith('X') and col not in['id', 'file_path', 'jpeg_bytes']]
        #self.target_cols = [col for col in self.data.columns if col.startswith('X') and col.endswith('_mean')]

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.X_jpegs[idx]))
        if self.transform:
            image = self.transform(image)

        # Get the input features and target variables separately
        input_data = self.X_train.iloc[idx][self.input_cols].values.astype(np.float32)
        target_data = self.y_train[idx]

        return image, input_data, target_data


