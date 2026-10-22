import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels,momentum=0.01)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels,momentum=0.01)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class GoRankResNet(nn.Module):
    """
    A ResNet-based model for ranking Go positions.
    Input: (B, 500, 19, 19)
    Output: (B, 10) - scores for the 10 rank classes
    """
    # 減少通道數以降低模型容量，並改用 Global Pooling 取代 Flatten 大全連接層以消除空間過擬合。
    def __init__(self, in_channels=500, num_channels=128, num_blocks=8, num_classes=10):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels,momentum=0.01)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(num_channels * 2, 128),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        max_pool = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = torch.cat([avg_pool, max_pool], dim=1)
        out = self.head(out)

        return out

class GoPlayerResNet(nn.Module):
    """
    A ResNet-based model to generate player embeddings.
    Input: (B, 17, 19, 19)
    Output: (B, 128) L2-normalized embedding
    """
    def __init__(self, in_channels=17, num_channels=128, num_blocks=16, embed_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, embed_dim)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        out = self.avg_pool(out) # Pooling        
        out = out.view(out.size(0), -1) # Flatten
        out = self.mlp_head(out) # MLP Head
        out = F.normalize(out, p=2, dim=1) # L2 Normalize the embedding (crucial for metric learning)
        return out
