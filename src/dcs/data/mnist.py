from typing import Tuple
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

def load_mnist(data_dir: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return train, test
