@echo off
echo Starting Mutual Fund Analyzer...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run Streamlit app
streamlit run portfolio_app.py

pause
