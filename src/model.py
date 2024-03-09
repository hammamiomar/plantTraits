import torch
import torch.nn as nn
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = gamma.unsqueeze(-1).unsqueeze(-1) * out + beta.unsqueeze(-1).unsqueeze(-1)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetFiLM(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(ResNetFiLM, self).__init__()
        
        # Load pretrained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # FiLMed residual blocks
        self.film_blocks = nn.ModuleList([
            ResidualBlock(num_features, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        ])
        
        # FiLM generator
        self.film_generator = nn.Sequential(
            nn.Linear(num_input_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * 128 * 4)  # 4 FiLMed blocks, each with 128 channels
        )
        
        # Final classifier
        self.conv1x1 = nn.Conv2d(128, 512, kernel_size=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_output_features)
    
    def forward(self, image, input_data):
        # Extract features from the image using pretrained ResNet
        features = self.resnet(image)
        
        # Generate FiLM parameters from input data
        film_params = self.film_generator(input_data)
        film_params = film_params.view(-1, 4, 128 * 2)
        gammas, betas = torch.split(film_params, 128, dim=2)
        
        # Pass features through FiLMed residual blocks
        for i, block in enumerate(self.film_blocks):
            features = block(features, gammas[:, i], betas[:, i])
        
        # Final classifier
        out = self.conv1x1(features)
        out = self.global_max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out