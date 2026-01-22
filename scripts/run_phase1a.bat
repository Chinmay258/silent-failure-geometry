@echo off
call .\.venv\Scripts\activate
python src\train.py
python src\extract_embeddings.py
python src\plots_phase1a.py
