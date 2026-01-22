import torch
import numpy as np

class OODCIFAR(torch.utils.data.Dataset):
    def __init__(self, dataset, sigma=0.2):
        self.dataset = dataset
        self.sigma = sigma

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        noise = torch.randn_like(x) * self.sigma
        x = torch.clamp(x + noise, 0, 1)
        return x, y
