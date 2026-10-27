@echo off
cd /d "%~dp0"
python scripts\train_local.py --algo all --steps 50000 --episodes 12
pause
