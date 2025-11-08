# PowerShell script to run the app
Write-Host "Starting Mutual Fund Analyzer..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run portfolio_app.py

# Keep window open if error occurs
Read-Host -Prompt "Press Enter to exit"
