from typing import Tuple
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def load_cifar10(data_dir: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm_test)
    return train, test
