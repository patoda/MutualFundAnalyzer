@echo off
setlocal

cd /d C:\w\mutualfundanalyzer

set "APP_RUNNING="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8501 .*LISTENING"') do set "APP_RUNNING=1"

if defined APP_RUNNING (
	echo Mutual Fund Analyzer is already running. Opening browser...
	start "" http://localhost:8501
	exit /b 0
)

if not exist ".venv\Scripts\activate.bat" (
	echo Virtual environment not found at .venv\Scripts\activate.bat
	echo Please run setup first.
	exit /b 1
)

echo Starting Mutual Fund Analyzer...
start "" http://localhost:8501
call .venv\Scripts\activate.bat
streamlit run portfolio_app.py --server.headless true --server.port 8501
