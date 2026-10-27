@echo off
cd /d "%~dp0"
python scripts\demo_local.py --view 3d --policy sac --scenario two_lane
if errorlevel 1 pause
