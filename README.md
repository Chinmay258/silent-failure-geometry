# Silent Failure Detection via Prediction Geometry

Reproducible code and experiments for "Detecting Silent Failure in Neural Networks via Prediction Geometry".

## Setup
1. Create & activate a virtual environment

(Windows):
python -m venv .venv
..venv\Scripts\activate

(Linux/macOS):
python3 -m venv .venv
source .venv/bin/activate


2. Install dependencies:
pip install -r requirements.txt


3. Run experiments (examples):
- Phase-0 (MNIST, fast):
python src/train.py
python src/extract_embeddings.py
python src/plots_phase1a.py


- Phase-1B (CIFAR spurious):
python src/train_phase1b.py
python src/extract_embeddings_phase1b.py
python src/plots_phase1b.py


- Phase-1C (ResNet + OOD):
python src/train_phase1c_resnet.py
python src/extract_embeddings_phase1c_resnet.py
python src/plots_phase1c.py


Or run everything in order:
python run_all.py



Results and plots are saved under `results/`.



