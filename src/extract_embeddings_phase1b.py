import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_cifar import CIFARCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor()
])

test_ds = datasets.CIFAR10(
    root=".",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

def extract(epoch):
    model = CIFARCNN().to(device)
    model.load_state_dict(torch.load(f"cifar_checkpoint_{epoch}.pt"))
    model.eval()

    embs = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, e = model(x, return_embedding=True)
            embs.append(e.cpu())

    return torch.cat(embs).numpy()

for e in range(15):
    np.save(f"cifar_emb_epoch_{e}.npy", extract(e))

print("CIFAR embeddings extracted successfully.")
