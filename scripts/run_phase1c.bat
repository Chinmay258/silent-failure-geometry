@echo off
call .\.venv\Scripts\activate

echo Running Phase-1C (ResNet + Robustness)
python src/train_phase1c_resnet.py
python src/extract_embeddings_phase1c_resnet.py
python src/plots_phase1c.py

echo Phase-1C complete
