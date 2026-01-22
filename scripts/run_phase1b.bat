@echo off
call .\.venv\Scripts\activate

echo Running Phase-1B (CIFAR + Spurious Correlation)
python src/train_phase1b.py
python src/extract_embeddings_phase1b.py
python src/plots_phase1b.py

echo Phase-1B complete
