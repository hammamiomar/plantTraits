import torch
import torch.nn as nn
import torchvision.models as models

class FiLMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, gamma, beta):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)

        x += self.shortcut(residual)
        x = self.relu(x)

        return x

class PlantModel(nn.Module):
    def __init__(self, num_ancillary_features, num_output_features):
        super(PlantModel, self).__init__()

        # Image feature extractor
        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

        # Ancillary data embedding
        self.ancillary_embedding = nn.Linear(num_ancillary_features, 128)

        # FiLM generator
        self.film_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * 128)
        )

        # FiLM blocks with residual connections
        self.film_block1 = FiLMBlock(num_features, 128)
        self.film_block2 = FiLMBlock(128, 128)

        # Fusion and prediction
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128 + 128, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_output_features)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, ancillary_data):
        # Image feature extraction
        image_features = self.feature_extractor(image)

        # Ancillary data embedding
        ancillary_embedding = self.ancillary_embedding(ancillary_data)

        # FiLM parameter generation
        film_params = self.film_generator(ancillary_embedding)
        gamma1, beta1 = torch.split(film_params, 128, dim=1)

        # FiLM blocks with residual connections
        modulated_features = self.film_block1(image_features, gamma1, beta1)
        gamma2, beta2 = torch.split(film_params, 128, dim=1)
        modulated_features = self.film_block2(modulated_features, gamma2, beta2)

        # Fusion and prediction
        pooled_features = self.global_avg_pool(modulated_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        fused_features = torch.cat((pooled_features, ancillary_embedding), dim=1)
        out = self.fc1(fused_features)
        out = self.relu(out)
        out = self.fc2(out)

        return out