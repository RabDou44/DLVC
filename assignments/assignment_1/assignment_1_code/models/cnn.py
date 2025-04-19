from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch




class YourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ## TODO implement
        self.conv_1 = torch.nn.Conv2d(
            3, 16, 5, stride=1,
            padding='same', padding_mode='zeros')
        self.bn_1 = torch.nn.BatchNorm2d(16)
        self.pool_1 = torch.nn.MaxPool2d(2)
        self.conv_2 = torch.nn.Conv2d(
            16, 32, 5, stride=1,
            padding='same', padding_mode='zeros')
        self.bn_2 = torch.nn.BatchNorm2d(32)
        self.pool_2 = torch.nn.MaxPool2d(2)
        self.drop_1 = torch.nn.Dropout(0.5)
        self.linear_1 = torch.nn.Linear(2048, 512)
        self.bn_3 = torch.nn.BatchNorm1d(512)
        self.linear_2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        ## TODO implement
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool_2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop_1(x)
        x = self.linear_1(x)
        x = self.bn_3(x)
        x = torch.nn.functional.relu(x)
        return self.linear_2(x)

    