import torch
import numpy as np

class SpuriousCIFAR(torch.utils.data.Dataset):
    def __init__(self, dataset, strength=0.9):
        self.dataset = dataset
        self.strength = strength

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if np.random.rand() < self.strength:
            x[:, :4, :4] = y / 10.0  # label-dependent patch

        return x, y
