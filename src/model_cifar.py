import torch.nn as nn
import torch.nn.functional as F

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 8 * 128, 256)  # embedding
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = F.relu(self.fc1(x))
        out = self.fc2(emb)

        if return_embedding:
            return out, emb
        return out
