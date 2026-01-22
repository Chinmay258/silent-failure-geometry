@echo off
call .\.venv\Scripts\activate

call scripts\run_phase1a.bat
call scripts\run_phase1b.bat
call scripts\run_phase1c.bat

echo All phases completed
