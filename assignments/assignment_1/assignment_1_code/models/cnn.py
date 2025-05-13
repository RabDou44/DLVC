from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, dropout_factor=0):
        super().__init__()

        # Convolutional block 1
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.bn_1 = nn.BatchNorm2d(16)
        self.pool_1 = nn.MaxPool2d(2)

        # Convolutional block 2
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.drop_1 = nn.Dropout2d(dropout_factor * 0.2)
        self.pool_2 = nn.MaxPool2d(2)

        # Convolutional block 3
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.drop_2 = nn.Dropout2d(dropout_factor * 0.3)

        # Fully connected layers
        self.linear_1 = nn.Linear(4096, 256)  # Adjust this if input image size changes
        self.bn_3 = nn.BatchNorm1d(256)
        self.drop_3 = nn.Dropout(dropout_factor * 0.5)

        self.linear_2 = nn.Linear(256, 64)
        self.linear_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pool_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.drop_1(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.drop_2(x)
        x = self.bn_2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.drop_3(x)
        x = self.bn_3(x)

        x = self.linear_2(x)
        x = F.relu(x)

        return self.linear_3(x)
