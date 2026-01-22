import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_cifar import CIFARCNN
from spurious_cifar import SpuriousCIFAR

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

EPOCHS = 15
BATCH = 64

transform = transforms.Compose([transforms.ToTensor()])

# -------------------------
# Data
# -------------------------
train_base = datasets.CIFAR10(".", train=True, download=True, transform=transform)
test_base  = datasets.CIFAR10(".", train=False, download=True, transform=transform)

train_ds = SpuriousCIFAR(train_base, strength=0.9)
test_ds  = test_base  # clean test

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256)

# -------------------------
# Model
# -------------------------
model = CIFARCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

acc_hist = []
confidence_history = []

# -------------------------
# Train
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0

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
    acc_hist.append(acc)
    mean_conf = torch.cat(confidences).mean().item()
    confidence_history.append(mean_conf)

    torch.save(model.state_dict(), f"cifar_checkpoint_{epoch}.pt")
    print(f"Epoch {epoch}: Accuracy = {acc:.4f}")

torch.save(acc_hist, "cifar_accuracy.pt")
torch.save(confidence_history, "confidence_1b.pt")
print("Training complete.")

