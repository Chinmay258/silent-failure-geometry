import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np

from model import SimpleCNN

# -------------------------
# Configuration
# -------------------------
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
LABEL_NOISE = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Label Noise Injection
# -------------------------
def add_label_noise(dataset, noise_ratio=0.05):
    n = len(dataset.targets)
    k = int(noise_ratio * n)
    idxs = random.sample(range(n), k)

    for i in idxs:
        dataset.targets[i] = random.randint(0, 9)

# -------------------------
# Data
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_ds = datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.MNIST(
    root=".",
    train=False,
    download=True,
    transform=transform
)

add_label_noise(train_ds, LABEL_NOISE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# -------------------------
# Model
# -------------------------
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

accuracy_history = []
confidence_history = []

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    # ---- Evaluation ----
    model.eval()
    correct = 0
    total = 0
    confidences = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            confidences.append(probs.max(dim=1)[0].cpu())

    acc = correct / total
    mean_conf = torch.cat(confidences).mean().item()

    accuracy_history.append(acc)
    confidence_history.append(mean_conf)

    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    print(
        f"Epoch {epoch}: "
        f"Accuracy = {acc:.4f}, "
        f"Confidence = {mean_conf:.4f}"
    )

# -------------------------
# Save Metrics
# -------------------------
torch.save(accuracy_history, "accuracy_1a.pt")
torch.save(confidence_history, "confidence_1a.pt")

print("Training complete.")
