# run_all.py - orchestrates phases and stores outputs in results/
import subprocess
import os
import datetime

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CHECKPOINTS, exist_ok=True)


def run(cmd):
    print(f"\n>>> Running: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=SRC)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(RESULTS, timestamp)
checkpoints = os.path.join(CHECKPOINTS, timestamp)
os.makedirs(outdir, exist_ok=True)

# PHASE-1A (MNIST)
run("python train.py")
run("python extract_embeddings.py")
run("python plots_phase1a.py")
# move artifacts
for fname in ["accuracy_1a.pt", "confidence_1a.pt"]:
    srcf = os.path.join(SRC, fname)
    if os.path.exists(srcf):
        os.replace(srcf, os.path.join(outdir, fname))
# move embeddings and figures
for f in os.listdir(SRC):
    if f.startswith("emb_epoch_") or f.endswith(".png"):
        os.replace(os.path.join(SRC, f), os.path.join(outdir, f))
    if f.startswith("checkpoint_epoch"):
        os.replace(os.path.join(SRC, f), os.path.join(checkpoints, f))

# PHASE-1B (CIFAR spurious)
run("python train_phase1b.py")
run("python extract_embeddings_phase1b.py")
run("python plots_phase1b.py")
# collect outputs
for f in os.listdir(SRC):
    if f.startswith("cifar_emb_epoch_") or f.endswith(".png") or f.startswith("confidence_1b.pt"):
        try:
            os.replace(os.path.join(SRC, f), os.path.join(outdir, f))
        except FileNotFoundError:
            pass
    if f.startswith("cifar_"):
        try:
            os.replace(os.path.join(SRC, f), os.path.join(checkpoints, f))
        except FileNotFoundError:
            pass

# PHASE-1C (ResNet + OOD)
run("python train_phase1c_resnet.py")
run("python extract_embeddings_phase1c_resnet.py")
run("python plots_phase1c.py")
# collect outputs
for f in os.listdir(SRC):
    if f.startswith("resnet_emb_epoch_") or f.endswith(".png") or f.startswith("confidence_1c.pt"):
        try:
            os.replace(os.path.join(SRC, f), os.path.join(outdir, f))
        except FileNotFoundError:
            pass
    if f.startswith("resnet_"):
        try:
            os.replace(os.path.join(SRC, f), os.path.join(checkpoints, f))
        except FileNotFoundError:
            pass

print(f"\nAll done.")
print(f"Results saved in: {outdir}")
print(f"Checkpoints saved in: {checkpoints}")
