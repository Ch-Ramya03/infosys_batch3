@echo off
cd /d "%~dp0"
echo Starting Air Quality Dashboard...
python -m streamlit run main_dashboard.py
pause
