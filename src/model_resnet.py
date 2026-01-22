import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Embed(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # embedding
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False):
        emb = self.backbone(x)
        out = self.fc(emb)
        if return_embedding:
            return out, emb
        return out
