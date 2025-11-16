@echo off
cd /d ""C:\Users\trive\Downloads\infosys\m3""
start "" http://localhost:8501
python -m streamlit run streamlit_app.py
pause
