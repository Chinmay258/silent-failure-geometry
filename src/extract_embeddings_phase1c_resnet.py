import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_resnet import ResNet18Embed

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

test_ds = datasets.CIFAR10(".", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=128)

def extract(epoch):
    model = ResNet18Embed().to(device)
    model.load_state_dict(torch.load(f"resnet_checkpoint_{epoch}.pt"))
    model.eval()

    embs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, e = model(x, return_embedding=True)
            embs.append(e.cpu())

    return torch.cat(embs).numpy()

for e in range(10):
    np.save(f"resnet_emb_epoch_{e}.npy", extract(e))

print("ResNet embeddings extracted.")
