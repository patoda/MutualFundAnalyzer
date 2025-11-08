#!/bin/bash

# Startup script for Azure Web App
python -m streamlit run portfolio_app.py --server.port 8000 --server.address 0.0.0.0
