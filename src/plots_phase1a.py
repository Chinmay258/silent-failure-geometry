import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from geometry import dispersion, effective_rank

# -------------------------
# Utilities
# -------------------------
def normalize(x):
    x = np.array(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def first_change(x, thresh=0.1):
    x = normalize(x)
    base = x[0]
    for i in range(1, len(x)):
        if abs(x[i] - base) > thresh:
            return i
    return None

# -------------------------
# Load Metrics
# -------------------------
acc = np.array(torch.load("accuracy_1a.pt"), dtype=np.float32)
conf = np.array(torch.load("confidence_1a.pt"), dtype=np.float32)

disp, rank = [], []

EPOCHS = len(acc)
for e in range(EPOCHS):
    E = np.load(f"emb_epoch_{e}.npy")
    disp.append(dispersion(E))
    rank.append(effective_rank(E))

disp = np.array(disp)
rank = np.array(rank)

# -------------------------
# Lead Time Analysis
# -------------------------
print("Lead Epochs:")
print("Accuracy  :", first_change(acc))
print("Confidence:", first_change(conf))
print("Dispersion:", first_change(disp))
print("Rank      :", first_change(rank))

# -------------------------
# Plot (Normalized)
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(normalize(acc), label="Accuracy (norm)")
plt.plot(normalize(conf), label="Confidence (norm)")
plt.plot(normalize(disp), label="Dispersion (norm)")
plt.plot(normalize(rank), label="Effective Rank (norm)")
plt.xlabel("Epoch")
plt.ylabel("Normalized Value")
plt.title("Phase-1A: Early Silent Failure Detection")
plt.legend()
plt.tight_layout()
plt.savefig("phase1a_results.png", dpi=300)
plt.show()
