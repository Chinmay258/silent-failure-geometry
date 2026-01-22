import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # embedding layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_embedding=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        emb = F.relu(self.fc1(x))
        out = self.fc2(emb)

        if return_embedding:
            return out, emb
        return out
