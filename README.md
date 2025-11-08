# Portfolio Tax Harvesting Analyzer

A comprehensive Streamlit web application for analyzing mutual fund portfolios and tax harvesting strategies.

## Features

- 📊 **Portfolio Overview** - Complete portfolio summary with XIRR calculations
- 📋 **All Holdings** - Detailed view of all schemes with LT/ST breakdown
- 💰 **Single-Scheme Tax Harvesting** - Calculate optimal units to sell for tax harvesting
- 🎯 **Multi-Fund Strategy** - Balanced tax harvesting across multiple funds
- 💳 **UPI Payment Integration** - Support the developer with QR code

## Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch (main), and main file path (portfolio_app.py)
   - Click "Deploy"

3. **Your app will be live at:** `https://<your-username>-<repo-name>.streamlit.app`

### Option 2: Local Deployment

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run portfolio_app.py
   ```

4. **Access locally:** `http://localhost:8501`

### Option 3: Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY portfolio_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "portfolio_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t portfolio-analyzer .
docker run -p 8501:8501 portfolio-analyzer
```

### Option 4: Heroku Deployment

1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run portfolio_app.py
```

3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

## Usage

1. Upload your CAS (Consolidated Account Statement) PDF
2. Enter your PDF password
3. Navigate through different tabs to:
   - View portfolio overview
   - Analyze individual holdings
   - Calculate tax harvesting strategies
   - Plan multi-fund distributions

## Technologies Used

- **Streamlit** - Web framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **PDFPlumber** - PDF parsing
- **SciPy** - XIRR calculations
- **QRCode** - UPI payment QR generation

## Security Note

- PDF files are processed in-memory and not stored
- Passwords are not logged or saved
- All data processing happens client-side

## Support

If you find this tool useful, consider supporting the developer via UPI: `ankitpatodiya@okicici`

## License

MIT License
