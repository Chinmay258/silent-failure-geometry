import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = datasets.MNIST(".", train=False, download=True,
                         transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=256)

def extract(epoch):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(f"checkpoint_epoch_{epoch}.pt"))
    model.eval()

    embs = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, e = model(x, return_embedding=True)
            embs.append(e.cpu())

    return torch.cat(embs).numpy()

for e in range(10):
    np.save(f"emb_epoch_{e}.npy", extract(e))
