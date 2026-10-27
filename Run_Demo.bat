@echo off
cd /d "%~dp0"
python scripts\demo_local.py --view 3d --policy sac --scenario traffic
if errorlevel 1 pause
