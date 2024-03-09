import torch
import numpy as np
import torch.nn as nn

def calculate_normalization_stats(dataset):
    image_data_all = []
    input_data_all = []
    
    for img, input_data, _ in dataset:
        image_data_all.append(torch.tensor(np.array(img)).permute(2, 0, 1))
        input_data_all.append(torch.tensor(input_data))
    
    image_data_all = torch.stack(image_data_all)
    input_data_all = torch.stack(input_data_all)
    
    image_mean = image_data_all.float().mean(dim=[0, 2, 3])
    image_std = image_data_all.float().std(dim=[0, 2, 3])
    input_data_mean = input_data_all.float().mean(dim=0)
    input_data_std = input_data_all.float().std(dim=0)
    
    return image_mean, image_std, input_data_mean, input_data_std

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
