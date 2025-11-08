"""
Interactive Portfolio Tax Harvesting Web App
Built with Streamlit - Run with: streamlit run portfolio_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import re
from collections import defaultdict
from scipy import optimize
import qrcode
from io import BytesIO
import base64
import tempfile
import os
import time

# Set page config
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CORE FUNCTIONS =====================

def generate_upi_qr(upi_id, amount=None):
    """Generate UPI QR code and return as base64 image"""
    # Create UPI payment string
    upi_string = f"upi://pay?pa={upi_id}"
    if amount:
        upi_string += f"&am={amount}"
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(upi_string)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def show_donation_banner():
    """Display donation banner with QR code and UPI deep link"""
    upi_id = "ankitpatodiya@okicici"
    
    st.markdown("---")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.2rem; border-radius: 1rem; color: white; text-align: center;">
            <h3 style="margin: 0; color: white;">☕ Buy me a treat :)</h3>
            <div style="background-color: rgba(255,255,255,0.2); padding: 0.6rem; 
                        border-radius: 0.5rem; margin-top: 0.8rem; font-family: monospace; font-size: 1.1rem;">
                <span id="upi-id">{upi_id}</span>
                <button onclick="copyUPI()" style="margin-left: 1rem; background-color: rgba(255,255,255,0.3); 
                        border: 1px solid rgba(255,255,255,0.5); color: white; padding: 0.3rem 0.8rem; 
                        border-radius: 0.3rem; cursor: pointer; font-size: 0.9rem;">
                    📋 Copy
                </button>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.9;">
                💳 Works with GPay, PhonePe, Paytm & all UPI apps
            </p>
        </div>
        
        <script>
        function copyUPI() {{
            const upiId = document.getElementById('upi-id').textContent;
            navigator.clipboard.writeText(upiId).then(function() {{
                // Show success message
                const btn = event.target;
                const originalText = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                btn.style.backgroundColor = 'rgba(76, 175, 80, 0.5)';
                setTimeout(function() {{
                    btn.innerHTML = originalText;
                    btn.style.backgroundColor = 'rgba(255,255,255,0.3)';
                }}, 2000);
            }});
        }}
        </script>
        """.format(upi_id=upi_id), unsafe_allow_html=True)
        
        # UPI deep link button for mobile users
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1rem;">
            <a href="upi://pay?pa={upi_id}&pn=Portfolio Analyzer&cu=INR" 
               style="display: inline-block; background-color: #4CAF50; color: white; 
                      padding: 0.8rem 2rem; text-decoration: none; border-radius: 0.5rem; 
                      font-weight: bold; font-size: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                📱 Pay via UPI App
            </a>
            <p style="font-size: 0.75rem; color: #666; margin-top: 0.5rem;">
                Tap to open GPay/PhonePe/Paytm (Mobile only)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Generate and display QR code
        qr_base64 = generate_upi_qr(upi_id)
        st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{qr_base64}" 
                 style="width: 150px; height: 150px; border: 3px solid #667eea; 
                        border-radius: 1rem; background: white; padding: 0.4rem;">
            <p style="margin-top: 0.5rem; font-weight: bold; color: #667eea; font-size: 0.9rem;">Scan to Pay</p>
        </div>
        """, unsafe_allow_html=True)

def format_indian_currency(value):
    """Format value in Indian numbering system (lakhs, crores)"""
    if value >= 10000000:  # 1 crore or more
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:  # 1 lakh or more
        return f"₹{value/100000:.2f} L"
    elif value >= 1000:  # 1 thousand or more
        return f"₹{value/1000:.2f} K"
    else:
        return f"₹{value:.2f}"

def format_indian_number(value):
    """Format number in Indian numbering system with proper comma placement."""
    # Handle negative numbers
    if value < 0:
        return '-' + format_indian_number(-value)
    
    # Convert to string and split on decimal point
    str_value = f"{value:.0f}"
    
    # Handle small numbers (less than 1000)
    if len(str_value) <= 3:
        return str_value
    
    # Indian numbering: last 3 digits, then groups of 2
    last_three = str_value[-3:]
    remaining = str_value[:-3]
    
    # Add commas every 2 digits from right to left
    result = last_three
    while remaining:
        if len(remaining) <= 2:
            result = remaining + ',' + result
            remaining = ''
        else:
            result = remaining[-2:] + ',' + result
            remaining = remaining[:-2]
    
    return result

def format_indian_rupee_compact(value):
    """
    Format rupees in Indian format for DataFrame display.
    Shows full Indian formatted number without decimals.
    """
    return format_indian_number(abs(value)) if value >= 0 else f"-{format_indian_number(abs(value))}"

def classify_fund_type(scheme_name):
    """Classify fund as equity/debt/international and return LTCG holding period."""
    scheme_lower = scheme_name.lower()
    
    # International/Foreign Equity funds (treated as debt for tax - 36 months)
    international_keywords = ['international', 'global', 'foreign', 'overseas', 'world', 
                             'us equity', 'nasdaq', 'emerging market']
    if any(keyword in scheme_lower for keyword in international_keywords):
        return 'international', 36 * 30  # 36 months in days
    
    # Debt funds (36 months LTCG)
    debt_keywords = ['liquid', 'ultra short', 'low duration', 'money market', 'overnight',
                    'gilt', 'debt', 'bond', 'income', 'credit', 'banking & psu',
                    'corporate bond', 'dynamic bond', 'floating rate', 'short duration',
                    'medium duration', 'long duration']
    if any(keyword in scheme_lower for keyword in debt_keywords):
        return 'debt', 36 * 30  # 36 months in days
    
    # Default: Equity (12 months LTCG)
    return 'equity', 12 * 30  # 12 months in days


def calculate_xirr(cashflows):
    """Calculate XIRR (annualized return) from cashflows. Returns percentage."""
    if len(cashflows) < 2:
        return 0
    
    dates = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    
    # Convert dates to days from first date
    first_date = min(dates)
    days = [(d - first_date).days for d in dates]
    years = [d / 365.0 for d in days]
    
    # Define NPV function
    def npv(rate):
        return sum(amt / ((1 + rate) ** yr) for amt, yr in zip(amounts, years))
    
    try:
        # Use brentq to find the rate where NPV = 0
        xirr = optimize.brentq(npv, -0.999, 10.0)
        return xirr * 100  # Convert to percentage
    except:
        return 0


def parse_pdf_cas(pdf_path, password):
    """Extract portfolio summary and all transactions from PDF."""
    
    with pdfplumber.open(pdf_path, password=password) as pdf:
        # Page 1: Portfolio summary
        page1 = pdf.pages[0].extract_text()
        
        amc_totals = {}
        for line in page1.split('\n'):
            match = re.search(r'(.+?(?:Mutual Fund|MF))\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)', line)
            if match:
                amc = match.group(1).strip()
                cost = float(match.group(2).replace(',', ''))
                market = float(match.group(3).replace(',', ''))
                amc_totals[amc] = {'cost': cost, 'market': market}
        
        # Extract all transactions AND current NAVs from closing balance lines
        transactions = []
        current_navs = {}  # Store current NAV per scheme
        current_folio = None
        current_scheme = None
        
        for page in pdf.pages:
            text = page.extract_text()
            
            for line in text.split('\n'):
                # Folio header
                folio_match = re.search(r'Folio No:\s*(\S+)', line)
                if folio_match:
                    current_folio = folio_match.group(1)
                    current_scheme = None
                    continue
                
                # Scheme name
                if current_folio and not current_scheme:
                    scheme_match = re.search(r'^[A-Z0-9]+-(.+?)(?:\s*\(Non-Demat\)|\s*\(Demat\)|\s*-\s*Registrar|\s*-\s*ISIN)', line)
                    if scheme_match:
                        current_scheme = scheme_match.group(1).strip()
                        continue
                
                # Extract current NAV from Closing Unit Balance line
                if current_scheme and 'Closing Unit Balance' in line:
                    # Pattern: "Closing Unit Balance: 17,962.047 NAV on 07-Nov-2025: INR 53.7342"
                    nav_match = re.search(r'NAV on \d{2}-[A-Za-z]{3}-\d{4}:\s*INR\s*([\d,]+\.?\d*)', line)
                    if nav_match:
                        nav_value = float(nav_match.group(1).replace(',', ''))
                        current_navs[current_scheme] = nav_value
                
                # Transaction line
                if not re.match(r'\d{2}-[A-Za-z]{3}-\d{4}', line):
                    continue
                
                if '***' in line or 'To 08-Nov-2025' in line:
                    continue
                
                # Extract date
                date_match = re.match(r'(\d{2}-[A-Za-z]{3}-\d{4})', line)
                date_str = date_match.group(1)
                
                try:
                    txn_date = datetime.strptime(date_str, '%d-%b-%Y')
                except:
                    continue
                
                # Extract numbers
                all_numbers = []
                for match in re.finditer(r'\(?([\d,]+\.?\d+)\)?', line):
                    num_str = match.group(1)
                    is_negative = match.group(0).startswith('(')
                    try:
                        value = float(num_str.replace(',', ''))
                        if value < 100000000 or value != int(value):
                            all_numbers.append((value, is_negative))
                    except:
                        continue
                
                if len(all_numbers) < 2:
                    continue
                
                # Get description
                rest = line[len(date_str):].strip()
                
                # Standard: last 4 numbers = Amount, Units, Price, Balance
                if len(all_numbers) >= 4:
                    amount_val, amount_neg = all_numbers[-4]
                    units_val, units_neg = all_numbers[-3]
                    price_val, _ = all_numbers[-2]
                    balance_val, _ = all_numbers[-1]
                    
                    amount = -amount_val if amount_neg else amount_val
                    units = -units_val if units_neg else units_val
                    price = price_val
                    balance = balance_val
                    
                    # Remove last 4 numbers from description
                    desc = rest
                    for _ in range(4):
                        desc = re.sub(r'\(?\d[\d,]*\.?\d*\)?$', '', desc).strip()
                else:
                    # Creation/segregation: 2 numbers = Units, Balance
                    units_val, units_neg = all_numbers[-2]
                    balance_val, _ = all_numbers[-1]
                    
                    units = -units_val if units_neg else units_val
                    balance = balance_val
                    amount = 0
                    price = 0
                    
                    desc = rest
                    for _ in range(2):
                        desc = re.sub(r'\(?\d[\d,]*\.?\d*\)?$', '', desc).strip()
                
                transactions.append({
                    'date': txn_date,
                    'scheme': current_scheme or 'Unknown',
                    'folio': current_folio or 'Unknown',
                    'description': desc,
                    'units': units,
                    'price': price,
                    'amount': amount,
                    'balance': balance
                })
    
    return pd.DataFrame(transactions), current_navs


def calculate_fifo_lots(transactions_df):
    """Calculate current holdings using FIFO method - track by scheme+folio."""
    
    lots = []
    
    # Group by BOTH scheme AND folio (each folio is a separate account)
    for (scheme, folio) in transactions_df.groupby(['scheme', 'folio']).groups.keys():
        scheme_folio_txns = transactions_df[
            (transactions_df['scheme'] == scheme) & 
            (transactions_df['folio'] == folio)
        ].sort_values('date')
        
        # Skip segregated portfolios (they're typically worthless)
        if 'segregat' in scheme.lower():
            continue
        
        # Check final balance - if zero, skip this folio
        final_balance = scheme_folio_txns.iloc[-1]['balance'] if len(scheme_folio_txns) > 0 else 0
        if final_balance < 0.01:
            continue
        
        # Track lots (purchases) and match with redemptions
        purchase_queue = []
        
        for _, txn in scheme_folio_txns.iterrows():
            if txn['units'] > 0:
                # Skip bonus/dividend/segregation transactions (zero cost but add units)
                if txn['amount'] == 0 or 'creation' in txn['description'].lower():
                    continue
                
                # Purchase - add to queue (including switch-ins, which have proper cost)
                purchase_queue.append({
                    'date': txn['date'],
                    'units': txn['units'],
                    'price': txn['price'] if txn['price'] > 0 else (txn['amount'] / txn['units'] if txn['units'] > 0 else 0),
                    'amount': txn['amount'],
                    'folio': folio
                })
            elif txn['units'] < 0:
                # Redemption (including switch-outs) - consume from queue (FIFO)
                units_to_redeem = abs(txn['units'])
                
                while units_to_redeem > 0.001 and purchase_queue:
                    lot = purchase_queue[0]
                    
                    if lot['units'] <= units_to_redeem:
                        # Fully redeem this lot
                        units_to_redeem -= lot['units']
                        purchase_queue.pop(0)
                    else:
                        # Partially redeem
                        lot['units'] -= units_to_redeem
                        lot['amount'] = lot['units'] * lot['price']
                        units_to_redeem = 0
        
        # Verify FIFO units match final balance
        fifo_total_units = sum(lot['units'] for lot in purchase_queue)
        
        # Allow small rounding difference
        if abs(fifo_total_units - final_balance) > 1.0:
            # Significant mismatch - likely due to bonus/corporate actions
            # Adjust proportionally
            if fifo_total_units > 0:
                adjustment_factor = final_balance / fifo_total_units
                for lot in purchase_queue:
                    lot['units'] *= adjustment_factor
                    lot['amount'] *= adjustment_factor
        
        # Remaining lots are current holdings
        for lot in purchase_queue:
            if lot['units'] > 0.001:
                lots.append({
                    'scheme': scheme,
                    'folio': lot['folio'],
                    'purchase_date': lot['date'],
                    'units': lot['units'],
                    'purchase_price': lot['price'],
                    'invested': lot['amount']
                })
    
    return pd.DataFrame(lots)


def calculate_cagr(invested, current_value, days):
    """Calculate CAGR (Compound Annual Growth Rate)"""
    if invested <= 0 or days <= 0:
        return 0
    years = days / 365.25
    if years < 0.01:  # Less than ~4 days
        return 0
    try:
        cagr = ((current_value / invested) ** (1 / years) - 1) * 100
        return cagr
    except:
        return 0


@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes, password):
    """Process PDF and return all calculated data. Results are cached based on file content and password."""
    
    try:
        # Write bytes to temp file for pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            pdf_path = tmp_file.name
        
        try:
            # Parse PDF - now returns transactions AND current NAVs from Closing Balance lines
            transactions_df, current_navs = parse_pdf_cas(pdf_path, password)
            
            if len(transactions_df) == 0:
                return None
            
            # Calculate current lots
            lots_df = calculate_fifo_lots(transactions_df)
            
            if len(lots_df) == 0:
                return None
            
        finally:
            # Clean up temp file
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass
        
    except Exception as e:
        return None
    
    # Use current NAVs from Closing Unit Balance lines (most accurate)
    # Fall back to transaction prices if NAV not found
    nav_dict = current_navs.copy()
    
    # For schemes without a closing balance NAV, use fallback logic
    for scheme in lots_df['scheme'].unique():
        if scheme not in nav_dict:
            # Try to find NAV from most recent transaction with price
            scheme_txns = transactions_df[transactions_df['scheme'] == scheme].sort_values('date', ascending=False)
            for _, txn in scheme_txns.iterrows():
                if txn['price'] > 0:
                    nav_dict[scheme] = txn['price']
                    break
            # If still not found, use default
            if scheme not in nav_dict:
                nav_dict[scheme] = 100
    
    # Add current NAV to lots
    lots_df['current_nav'] = lots_df['scheme'].map(nav_dict)
    lots_df['current_value'] = lots_df['units'] * lots_df['current_nav']
    lots_df['gain_loss'] = lots_df['current_value'] - lots_df['invested']
    lots_df['gain_pct'] = lots_df['gain_loss'] / lots_df['invested']
    
    # Calculate holding days
    lots_df['holding_days'] = (datetime.now() - lots_df['purchase_date']).dt.days
    
    # Calculate CAGR for each lot
    lots_df['cagr'] = lots_df.apply(
        lambda row: calculate_cagr(row['invested'], row['current_value'], row['holding_days']),
        axis=1
    )
    
    # Classify fund types and LT/ST
    lots_df['fund_type'], lots_df['ltcg_days'] = zip(*lots_df['scheme'].apply(classify_fund_type))
    lots_df['is_lt'] = lots_df['holding_days'] > lots_df['ltcg_days']
    lots_df['lt_st'] = lots_df['is_lt'].apply(lambda x: 'LT' if x else 'ST')
    
    return {
        'lots_df': lots_df,
        'transactions_df': transactions_df,
        'nav_dict': nav_dict
    }

# ===================== END CORE FUNCTIONS =====================

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    
    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        line-height: 1 !important;
    }
    [data-testid="stMetric"] {
        padding: 0.3rem 0.5rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }
    
    /* Aggressive dataframe styling - force smaller text */
    div[data-testid="stDataFrame"] {
        font-size: 10px !important;
    }
    div[data-testid="stDataFrame"] div {
        font-size: 10px !important;
    }
    div[data-testid="stDataFrame"] table {
        font-size: 10px !important;
    }
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th {
        font-size: 10px !important;
        padding: 2px 4px !important;
        white-space: nowrap !important;
    }
    /* Target specific cell content */
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        font-size: 10px !important;
        padding: 2px 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">💼 Portfolio Analyzer</div>', unsafe_allow_html=True)

# Initialize session state for active tab if not exists
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Ensure nav_radio matches active_tab when active_tab is changed programmatically
# This prevents the radio button from showing the wrong selection
if 'nav_radio' in st.session_state and st.session_state.nav_radio != st.session_state.active_tab:
    # active_tab was changed programmatically, sync nav_radio
    st.session_state.nav_radio = st.session_state.active_tab

# Initialize session state for processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'ltcg_budget' not in st.session_state:
    st.session_state.ltcg_budget = 100000

# Initialize file tracking
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None

# Tab names - Landing page is first
tab_names = ["🏠 Landing Page", "📊 Portfolio Overview", "📋 All Holdings", "💰 Single-Scheme Tax Harvest", "🎯 Multi-Fund Strategy"]

# Callback to update active tab when user clicks
def on_tab_change():
    st.session_state.active_tab = st.session_state.nav_radio

# Radio button for navigation with callback
selected_tab = st.radio(
    "Navigation",
    range(len(tab_names)),
    format_func=lambda x: tab_names[x],
    index=st.session_state.active_tab,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
    on_change=on_tab_change
)

# Use session state value for rendering (this is the source of truth)
active_tab = st.session_state.active_tab

# LANDING PAGE TAB
if active_tab == 0:
    st.header("📁 Upload Your CAS Statement")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_pdf = st.file_uploader("**STEP 1: Upload PDF CAS**", type=['pdf'], help="Click 'Browse files' button to select your CAS PDF")
        
        # Track file changes - clear processed data ONLY when a different file is uploaded
        if uploaded_pdf is not None:
            current_file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
            
            # Check if this is a different file than before
            if 'last_uploaded_file_id' in st.session_state:
                if st.session_state.last_uploaded_file_id != current_file_id:
                    # Different file uploaded - clear old data
                    st.session_state.processed_data = None
                    st.session_state.last_uploaded_file_id = current_file_id
                    st.info(f"📎 New file detected: **{uploaded_pdf.name}**")
            else:
                # First file upload
                st.session_state.last_uploaded_file_id = current_file_id
        
    with col2:
        password = st.text_input("**STEP 2: Enter Password**", type="password", help="Enter the password from your email (leave blank if no password)")
    
    # Process button
    process_clicked = False
    if uploaded_pdf:
        process_clicked = st.button("**STEP 3: Process PDF** 🚀", type="primary", use_container_width=True)
    
    # Process PDF when button is clicked - RIGHT BELOW THE BUTTON
    if process_clicked and uploaded_pdf:
        with st.spinner('🔄 Processing PDF CAS... This may take a minute...'):
            # Reset file pointer to beginning
            uploaded_pdf.seek(0)
            
            # Read file bytes
            pdf_bytes = uploaded_pdf.read()
            
            try:
                # Process PDF - use None if password is empty
                pdf_password = password if password else None
                data = process_pdf(pdf_bytes, pdf_password)
                
                # Check what we got
                if data is None:
                    st.error("❌ **📄 No valid CAS data found! This doesn't appear to be a valid Consolidated Account Statement. Please ensure you uploaded the correct PDF.**")
                elif 'lots_df' not in data or len(data['lots_df']) == 0:
                    st.error("❌ **📄 No holdings found in CAS! The PDF was processed but no current holdings were found. You may have zero balance in all schemes.**")
                elif 'transactions_df' not in data or len(data['transactions_df']) == 0:
                    st.error("❌ **📄 No transactions found in CAS! The PDF format might not be supported.**")
                else:
                    # Store in session state
                    st.session_state.processed_data = data
                    
                    # Show success
                    st.success(f"✅ **Success!** Loaded {len(data['transactions_df'])} transactions, {data['lots_df']['scheme'].nunique()} schemes, {len(data['lots_df'])} lots")
                    
                    # Switch to Portfolio Overview tab (index 1)
                    st.session_state.active_tab = 1
                    
                    # Rerun to switch tabs
                    st.rerun()
                    
            except Exception as e:
                # Handle any errors during processing
                error_msg = str(e)
                exception_type = type(e).__name__
                
                # Check for password-related errors
                password_keywords = [
                    "password", "decrypt", "encrypted", "authentication", 
                    "incorrect password", "file has not been decrypted",
                    "pdferror", "owner password", "user password",
                    "not allowed", "encrypted pdf", "permissionerror"
                ]
                
                # Check for invalid format errors
                format_keywords = [
                    "index", "list index", "keyerror", "attributeerror",
                    "nonetype", "pages", "extract_text"
                ]
                
                # Determine error type
                is_password_error = (
                    exception_type in ["PdfminerException", "PermissionError", "PdfReadError"] or
                    any(keyword in error_msg.lower() for keyword in password_keywords) or
                    any(keyword in exception_type.lower() for keyword in ["permission", "password", "decrypt"]) or
                    (exception_type == "PdfminerException" and len(error_msg.strip()) < 5)
                )
                
                is_format_error = (
                    exception_type in ["IndexError", "KeyError", "AttributeError", "TypeError"] or
                    any(keyword in error_msg.lower() for keyword in format_keywords)
                )
                
                if is_password_error:
                    st.error("❌ **🔒 WRONG PASSWORD!** The password you entered is incorrect. Please check your email for the correct password.")
                    st.warning("💡 **Tip:** Check your email for the correct password. It's usually sent along with the CAS PDF link.")
                elif is_format_error:
                    st.error("❌ **📄 Invalid File Format!** This doesn't appear to be a valid CAS (Consolidated Account Statement) PDF. Please upload a CAS statement from CAMS/Karvy.")
                else:
                    st.error(f"❌ **⚠️ Error:** {exception_type} - {error_msg if error_msg else 'Unknown error'}")
    
    st.markdown("---")
    
    # Show appropriate content on landing page
    if st.session_state.processed_data is not None and not process_clicked:
        # Data has been processed and we're back on landing page
        st.success("✅ **Portfolio data loaded!** Navigate to other tabs to view analysis.")
        
        data = st.session_state.processed_data
        st.info(f"📊 Loaded: {len(data['transactions_df'])} transactions | {data['lots_df']['scheme'].nunique()} schemes | {len(data['lots_df'])} lots")
        
        st.markdown("**Ready to explore:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Portfolio Overview", use_container_width=True, type="primary"):
                st.session_state.active_tab = 1
                st.rerun()
        with col2:
            if st.button("📋 All Holdings", use_container_width=True):
                st.session_state.active_tab = 2
                st.rerun()
        with col3:
            if st.button("💰 Tax Harvest", use_container_width=True):
                st.session_state.active_tab = 3
                st.rerun()
        with col4:
            if st.button("🎯 Multi-Fund", use_container_width=True):
                st.session_state.active_tab = 4
                st.rerun()
        
        # Show donation banner at bottom
        show_donation_banner()
    else:
        # Show welcome screen when no file uploaded or currently processing
        st.subheader("Welcome! 👋")
        
        st.write("This tool helps you analyze your mutual fund portfolio and optimize tax harvesting strategies.")
        
        st.info("""
        **How to get your CAS (Consolidated Account Statement):**
        
        1. Visit [CAMS website](https://www.camsonline.com/Investors/Statements/Consolidated-Account-Statement)
        2. Enter your email ID registered with mutual funds
        3. Select "Detailed" statement type
        4. Choose date range (e.g., Since Inception)
        5. Select "Password Protected" PDF format
        6. Submit and check your email
        7. You'll receive the PDF with password in the email
        8. Upload that PDF here!
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **Features:**
        - 📊 Complete portfolio analysis with XIRR calculations
        - 💰 Tax harvesting recommendations (LTCG optimization)
        - 📈 Scheme-wise performance tracking
        - 🎯 Multi-fund balanced harvesting strategies
        """)
        
        # Show donation banner at bottom
        show_donation_banner()

# OTHER TABS - Use data from session state
elif active_tab > 0:
    # Check if data is available
    if st.session_state.processed_data is None:
        st.warning("⚠️ No portfolio data loaded. Please go to the Landing Page to upload your CAS PDF.")
        if st.button("← Go to Landing Page", key="goto_landing"):
            st.session_state.active_tab = 0
            st.rerun()
    else:
        # Extract data from session state
        data = st.session_state.processed_data
        lots_df = data['lots_df']
        transactions_df = data['transactions_df']
        nav_dict = data['nav_dict']
        ltcg_budget = st.session_state.ltcg_budget
        
        # Show quick info bar at top
        st.info(f"📊 Loaded: {len(transactions_df)} transactions | {lots_df['scheme'].nunique()} schemes | {len(lots_df)} lots | LTCG Budget: ₹{format_indian_number(ltcg_budget)}")
        
        st.markdown("---")
        
        # Display content based on selected tab
        if active_tab == 1:
            # TAB 1: Portfolio Overview
            st.header("Portfolio Overview")
            
            # Calculate summary metrics
            total_invested = lots_df['invested'].sum()
            total_value = lots_df['current_value'].sum()
            total_gain = total_value - total_invested
            gain_pct = (total_gain / total_invested) if total_invested > 0 else 0
            
            lt_lots = lots_df[lots_df['is_lt']]
            st_lots = lots_df[~lots_df['is_lt']]
            
            lt_invested = lt_lots['invested'].sum()
            lt_value = lt_lots['current_value'].sum()
            lt_gain = lt_value - lt_invested
            
            st_invested = st_lots['invested'].sum()
            st_value = st_lots['current_value'].sum()
            st_gain = st_value - st_invested
            
            num_schemes = lots_df['scheme'].nunique()
            num_lots = len(lots_df)
            
            # Calculate XIRR for total portfolio
            total_cashflows = []
            for _, lot in lots_df.iterrows():
                total_cashflows.append((lot['purchase_date'], -lot['invested']))
            total_cashflows.append((datetime.now(), total_value))
            total_xirr = calculate_xirr(total_cashflows)
            
            # Calculate XIRR for LT holdings
            lt_cashflows = []
            for _, lot in lt_lots.iterrows():
                lt_cashflows.append((lot['purchase_date'], -lot['invested']))
            lt_cashflows.append((datetime.now(), lt_value))
            lt_xirr = calculate_xirr(lt_cashflows)
            
            # Calculate XIRR for ST holdings
            st_cashflows = []
            for _, lot in st_lots.iterrows():
                st_cashflows.append((lot['purchase_date'], -lot['invested']))
            st_cashflows.append((datetime.now(), st_value))
            st_xirr = calculate_xirr(st_cashflows)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"₹{format_indian_number(total_invested)}")
            with col2:
                st.metric("Current Value", f"₹{format_indian_number(total_value)}")
            with col3:
                st.metric("Total Gain", f"₹{format_indian_number(total_gain)}", delta=f"{gain_pct*100:.2f}%")
            with col4:
                st.metric("XIRR", f"{total_xirr:.2f}%", help="Portfolio-wide annualized return")
            
            st.markdown("---")
            
            # LT vs ST breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Long-Term Holdings")
                st.markdown(f"""
                <div class="success-card">
                    <b>Invested:</b> ₹{format_indian_number(lt_invested)}<br>
                    <b>Current Value:</b> ₹{format_indian_number(lt_value)}<br>
                    <b>Gain:</b> ₹{format_indian_number(lt_gain)} ({(lt_gain/lt_invested*100) if lt_invested > 0 else 0:.2f}%)<br>
                    <b>XIRR:</b> {lt_xirr:.2f}%<br>
                    <b>Lots:</b> {len(lt_lots)}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Short-Term Holdings")
                st.markdown(f"""
                <div class="warning-card">
                    <b>Invested:</b> ₹{format_indian_number(st_invested)}<br>
                    <b>Current Value:</b> ₹{format_indian_number(st_value)}<br>
                    <b>Gain:</b> ₹{format_indian_number(st_gain)} ({(st_gain/st_invested*100) if st_invested > 0 else 0:.2f}%)<br>
                    <b>XIRR:</b> {st_xirr:.2f}%<br>
                    <b>Lots:</b> {len(st_lots)}
                </div>
                """, unsafe_allow_html=True)
            
            # Pie chart
            st.markdown("---")
            st.subheader("LT vs ST Distribution")
            
            fig = go.Figure(data=[go.Pie(
                labels=['Long-Term', 'Short-Term'],
                values=[lt_value, st_value],
                hole=0.4,
                marker_colors=['#28a745', '#ffc107']
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top holdings
            st.markdown("---")
            st.subheader("Top 10 Holdings by Value")
            
            scheme_summary = lots_df.groupby('scheme').agg({
                'current_value': 'sum',
                'invested': 'sum'
            }).reset_index()
            scheme_summary['gain'] = scheme_summary['current_value'] - scheme_summary['invested']
            scheme_summary = scheme_summary.sort_values('current_value', ascending=False).head(10)
            
            # Shorten scheme names for visualization
            scheme_summary['short_name'] = scheme_summary['scheme'].apply(
                lambda x: x[:40] + '...' if len(x) > 40 else x
            )
            
            # Add formatted hover text
            scheme_summary['value_text'] = scheme_summary['current_value'].apply(format_indian_currency)
            scheme_summary['gain_text'] = scheme_summary['gain'].apply(format_indian_currency)
            
            fig = px.bar(
                scheme_summary,
                y='short_name',
                x='current_value',
                color='gain',
                orientation='h',
                title='Top 10 Schemes by Current Value',
                labels={'current_value': 'Value (₹)', 'gain': 'Gain (₹)', 'short_name': 'Scheme'},
                color_continuous_scale='RdYlGn',
                height=500,
                hover_data={
                    'current_value': False,
                    'gain': False,
                    'value_text': True,
                    'gain_text': True,
                    'short_name': False
                }
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Value: %{customdata[0]}<br>Gain: %{customdata[1]}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 2:
            # TAB 2: All Holdings
            st.header("All Holdings")
            
            # Initialize session state for this filter
            if 'all_holdings_scheme' not in st.session_state:
                st.session_state.all_holdings_scheme = "All Schemes"
            
            # Single filter for scheme selection
            schemes = sorted(lots_df['scheme'].unique())
            selected_scheme = st.selectbox(
                "Select Scheme", 
                ["All Schemes"] + schemes,
                index=(["All Schemes"] + schemes).index(st.session_state.all_holdings_scheme) if st.session_state.all_holdings_scheme in ["All Schemes"] + schemes else 0,
                key='all_holdings_scheme',
                help="Choose a specific scheme to see detailed breakdown"
            )
            
            if selected_scheme == "All Schemes":
                # Show summary table of all schemes
                st.subheader("📊 Portfolio Summary by Scheme")
                
                scheme_summary = []
                for scheme in sorted(lots_df['scheme'].unique()):
                    scheme_lots = lots_df[lots_df['scheme'] == scheme]
                    scheme_txns = transactions_df[transactions_df['scheme'] == scheme]
                    
                    total_units = scheme_lots['units'].sum()
                    total_invested = scheme_lots['invested'].sum()
                    total_value = scheme_lots['current_value'].sum()
                    total_gain = total_value - total_invested
                    gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0
                    
                    fund_type = scheme_lots.iloc[0]['fund_type'].upper()
                    
                    # Calculate XIRR for scheme
                    cashflows = []
                    for _, txn in scheme_txns.iterrows():
                        if txn['amount'] != 0:
                            cashflows.append((txn['date'], -txn['amount']))
                    if total_value > 0:
                        cashflows.append((datetime.now(), total_value))
                    xirr = calculate_xirr(cashflows) if len(cashflows) > 1 else 0
                    
                    # LT/ST breakdown
                    lt_lots = scheme_lots[scheme_lots['is_lt']]
                    st_lots = scheme_lots[~scheme_lots['is_lt']]
                    
                    lt_units = lt_lots['units'].sum()
                    st_units = st_lots['units'].sum()
                    
                    scheme_summary.append({
                        'Scheme': scheme,
                        'Fund Type': fund_type,
                        'Total Units': total_units,
                        'LT Units': lt_units,
                        'ST Units': st_units,
                        'Invested': total_invested,
                        'Current Value': total_value,
                        'Gain/Loss': total_gain,
                        'Gain %': gain_pct,
                        'XIRR': xirr
                    })
                
                summary_df = pd.DataFrame(scheme_summary)
                
                # Shorten scheme names and format for display
                display_df = summary_df.copy()
                display_df['Scheme'] = display_df['Scheme'].apply(lambda x: x[:45] + '...' if len(x) > 45 else x)
                display_df['Type'] = display_df['Fund Type']
                display_df['Units'] = display_df['Total Units'].apply(lambda x: f"{x:.0f}")
                display_df['LT'] = display_df['LT Units'].apply(lambda x: f"{x:.0f}")
                display_df['ST'] = display_df['ST Units'].apply(lambda x: f"{x:.0f}")
                display_df['Invested₹'] = display_df['Invested'].apply(lambda x: format_indian_rupee_compact(x))
                display_df['Value₹'] = display_df['Current Value'].apply(lambda x: format_indian_rupee_compact(x))
                display_df['Gain₹'] = display_df['Gain/Loss'].apply(lambda x: format_indian_rupee_compact(x))
                display_df['Gain%'] = display_df['Gain %'].apply(lambda x: f"{x:.1f}%")
                display_df['XIRR%'] = display_df['XIRR'].apply(lambda x: f"{x:.1f}%")
                
                final_df = display_df[['Scheme', 'Type', 'Units', 'LT', 'ST', 'Invested₹', 'Value₹', 'Gain₹', 'Gain%', 'XIRR%']]
                
                st.dataframe(
                    final_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                    column_config={
                        "Scheme": st.column_config.TextColumn("Scheme", width="large"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Units": st.column_config.TextColumn("Units", width="small"),
                        "LT": st.column_config.TextColumn("LT", width="small"),
                        "ST": st.column_config.TextColumn("ST", width="small"),
                        "Invested₹": st.column_config.TextColumn("Invested₹", width="medium"),
                        "Value₹": st.column_config.TextColumn("Value₹", width="medium"),
                        "Gain₹": st.column_config.TextColumn("Gain₹", width="medium"),
                        "Gain%": st.column_config.TextColumn("Gain%", width="small"),
                        "XIRR%": st.column_config.TextColumn("XIRR%", width="small"),
                    }
                )
                
                st.info("💡 Select a specific scheme from the dropdown above to see detailed lot-level breakdown")
                
            else:
                # Show detailed view for selected scheme
                scheme_lots = lots_df[lots_df['scheme'] == selected_scheme].copy()
                
                if len(scheme_lots) == 0:
                    st.warning(f"No holdings found for {selected_scheme}")
                else:
                    # Get fund info
                    fund_type = scheme_lots.iloc[0]['fund_type'].upper()
                    current_nav = scheme_lots.iloc[0]['current_nav']
                    
                    st.markdown(f"### {selected_scheme}")
                    st.markdown(f"**Fund Type:** {fund_type} | **Current NAV:** ₹{current_nav:.2f}")
                    
                    st.markdown("---")
                    
                    # Calculate aggregates
                    total_units = scheme_lots['units'].sum()
                    total_invested = scheme_lots['invested'].sum()
                    total_value = scheme_lots['current_value'].sum()
                    total_gain = total_value - total_invested
                    gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0
                    
                    # Calculate XIRR for total scheme
                    scheme_txns = transactions_df[transactions_df['scheme'] == selected_scheme]
                    cashflows = []
                    for _, txn in scheme_txns.iterrows():
                        if txn['amount'] != 0:
                            cashflows.append((txn['date'], -txn['amount']))
                    if total_value > 0:
                        cashflows.append((datetime.now(), total_value))
                    total_xirr = calculate_xirr(cashflows) if len(cashflows) > 1 else 0
                    
                    lt_lots = scheme_lots[scheme_lots['is_lt']]
                    st_lots = scheme_lots[~scheme_lots['is_lt']]
                    
                    # === SCHEME SUMMARY (Total) ===
                    with st.container():
                        st.markdown("#### 📊 SCHEME SUMMARY")
                        cols = st.columns(7)
                        cols[0].metric("Total Units", f"{total_units:.0f}")
                        cols[1].metric("Current NAV", f"₹{current_nav:.2f}")
                        cols[2].metric("Invested", f"₹{format_indian_number(total_invested)}")
                        cols[3].metric("Value", f"₹{format_indian_number(total_value)}")
                        cols[4].metric("Gain/Loss", f"₹{format_indian_number(total_gain)}")
                        cols[5].metric("Gain %", f"{gain_pct:.2f}%")
                        cols[6].metric("XIRR %", f"{total_xirr:.2f}%")
                    
                    st.markdown("---")
                    
                    # === LT SUMMARY ===
                    if len(lt_lots) > 0:
                        lt_units = lt_lots['units'].sum()
                        lt_invested = lt_lots['invested'].sum()
                        lt_value = lt_lots['current_value'].sum()
                        lt_gain = lt_value - lt_invested
                        lt_gain_pct = (lt_gain / lt_invested * 100) if lt_invested > 0 else 0
                        
                        # Calculate XIRR for LT lots only
                        lt_purchase_dates = set(lt_lots['purchase_date'].dt.date)
                        lt_cashflows = []
                        for _, txn in scheme_txns.iterrows():
                            if txn['amount'] != 0 and txn['date'].date() in lt_purchase_dates:
                                lt_cashflows.append((txn['date'], -txn['amount']))
                        if lt_value > 0:
                            lt_cashflows.append((datetime.now(), lt_value))
                        lt_xirr = calculate_xirr(lt_cashflows) if len(lt_cashflows) > 1 else 0
                        
                        with st.expander("🟢 **LONG TERM SUMMARY**", expanded=True):
                            st.markdown("##### Long-Term Holdings")
                            cols = st.columns(7)
                            cols[0].metric("LT Units", f"{lt_units:.0f}")
                            cols[1].metric("Lots", f"{len(lt_lots)}")
                            cols[2].metric("Invested", f"₹{format_indian_number(lt_invested)}")
                            cols[3].metric("Value", f"₹{format_indian_number(lt_value)}")
                            cols[4].metric("Gain/Loss", f"₹{format_indian_number(lt_gain)}")
                            cols[5].metric("Gain %", f"{lt_gain_pct:.2f}%")
                            cols[6].metric("XIRR %", f"{lt_xirr:.2f}%")
                            
                            st.markdown("---")
                            st.markdown("##### 📋 Individual Lots (FIFO Order)")
                            
                            # Show LT lots - compact columns
                            lt_display = lt_lots.sort_values('purchase_date').copy()
                            lt_display_df = pd.DataFrame({
                                'Date': lt_display['purchase_date'].dt.strftime('%d-%b-%y'),
                                'Units': lt_display['units'].apply(lambda x: f"{x:.1f}"),
                                'Buy': lt_display['purchase_price'].apply(lambda x: f"{x:.1f}"),
                                'Inv₹': lt_display['invested'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Val₹': lt_display['current_value'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Gain₹': lt_display['gain_loss'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Gain%': lt_display['gain_pct'].apply(lambda x: f"{x*100:.1f}%"),
                                'CAGR%': lt_display['cagr'].apply(lambda x: f"{x:.1f}%"),
                                'Days': lt_display['holding_days'].apply(lambda x: f"{x:.0f}")
                            })
                            
                            st.dataframe(
                                lt_display_df,
                                hide_index=True
                            )
                    
                    # === ST SUMMARY ===
                    if len(st_lots) > 0:
                        st_units = st_lots['units'].sum()
                        st_invested = st_lots['invested'].sum()
                        st_value = st_lots['current_value'].sum()
                        st_gain = st_value - st_invested
                        st_gain_pct = (st_gain / st_invested * 100) if st_invested > 0 else 0
                        
                        # Calculate XIRR for ST lots only
                        st_purchase_dates = set(st_lots['purchase_date'].dt.date)
                        st_cashflows = []
                        for _, txn in scheme_txns.iterrows():
                            if txn['amount'] != 0 and txn['date'].date() in st_purchase_dates:
                                st_cashflows.append((txn['date'], -txn['amount']))
                        if st_value > 0:
                            st_cashflows.append((datetime.now(), st_value))
                        st_xirr = calculate_xirr(st_cashflows) if len(st_cashflows) > 1 else 0
                        
                        with st.expander("🟠 **SHORT TERM SUMMARY**", expanded=True):
                            st.markdown("##### Short-Term Holdings")
                            cols = st.columns(7)
                            cols[0].metric("ST Units", f"{st_units:.0f}")
                            cols[1].metric("Lots", f"{len(st_lots)}")
                            cols[2].metric("Invested", f"₹{format_indian_number(st_invested)}")
                            cols[3].metric("Value", f"₹{format_indian_number(st_value)}")
                            cols[4].metric("Gain/Loss", f"₹{format_indian_number(st_gain)}")
                            cols[5].metric("Gain %", f"{st_gain_pct:.2f}%")
                            cols[6].metric("XIRR %", f"{st_xirr:.2f}%")
                            
                            st.markdown("---")
                            st.markdown("##### 📋 Individual Lots (FIFO Order)")
                            
                            # Show ST lots - compact columns
                            st_display = st_lots.sort_values('purchase_date').copy()
                            st_display_df = pd.DataFrame({
                                'Date': st_display['purchase_date'].dt.strftime('%d-%b-%y'),
                                'Units': st_display['units'].apply(lambda x: f"{x:.1f}"),
                                'Buy': st_display['purchase_price'].apply(lambda x: f"{x:.1f}"),
                                'Inv₹': st_display['invested'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Val₹': st_display['current_value'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Gain₹': st_display['gain_loss'].apply(lambda x: format_indian_rupee_compact(x)),
                                'Gain%': st_display['gain_pct'].apply(lambda x: f"{x*100:.1f}%"),
                                'CAGR%': st_display['cagr'].apply(lambda x: f"{x:.1f}%"),
                                'Days': st_display['holding_days'].apply(lambda x: f"{x:.0f}")
                            })
                            
                            st.dataframe(
                                st_display_df,
                                hide_index=True
                            )
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 3:
            # TAB 3: Single-Scheme Tax Harvest
            st.header("Single-Scheme Tax Harvesting")
            
            # LTCG Budget Setting
            st.subheader("⚙️ LTCG Budget")
            col1, col2 = st.columns([3, 1])
            with col1:
                ltcg_budget_input = st.number_input(
                    "**LTCG Budget (₹)**",
                    min_value=5000,
                    max_value=500000,
                    value=st.session_state.ltcg_budget,
                    step=5000,
                    help="Total Long-Term Capital Gains budget for tax harvesting per scheme",
                    key="ltcg_input_single"
                )
                # Update session state
                st.session_state.ltcg_budget = ltcg_budget_input
                ltcg_budget = ltcg_budget_input
            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
            
            st.info("💡 **What is LTCG Budget?** This is the maximum long-term profit you want to realize by selling units. The tool will show which units to sell to achieve this target profit.")
            
            st.markdown("---")
            
            st.info(f"Each scheme shows units to sell for ₹{format_indian_number(ltcg_budget)} LTCG (individual scheme basis)")
            
            # Get equity schemes with LT holdings
            equity_lt_schemes = lots_df[(lots_df['fund_type'] == 'equity') & (lots_df['is_lt'])].groupby('scheme').agg({
                'units': 'sum',
                'gain_loss': 'sum',
                'current_value': 'sum'
            }).reset_index()
            equity_lt_schemes = equity_lt_schemes.sort_values('gain_loss', ascending=False)
            
            if len(equity_lt_schemes) == 0:
                st.warning("No equity schemes with long-term holdings found")
            else:
                # Initialize session state for this filter
                if 'tax_harvest_scheme' not in st.session_state:
                    st.session_state.tax_harvest_scheme = "All Equity Schemes"
                
                # Add scheme filter
                scheme_list = sorted(equity_lt_schemes['scheme'].tolist())
                selected_harvest_scheme = st.selectbox(
                    "Select Scheme for Tax Harvesting",
                    ["All Equity Schemes"] + scheme_list,
                    index=(["All Equity Schemes"] + scheme_list).index(st.session_state.tax_harvest_scheme) if st.session_state.tax_harvest_scheme in ["All Equity Schemes"] + scheme_list else 0,
                    key='tax_harvest_scheme',
                    help="Choose a specific scheme or view all equity schemes with LT holdings"
                )
                
                if selected_harvest_scheme == "All Equity Schemes":
                    # Show summary table
                    st.subheader("📊 Tax Harvesting Summary")
                    
                    summary_data = []
                    
                    for idx, scheme_row in equity_lt_schemes.iterrows():
                        scheme = scheme_row['scheme']
                        
                        # Get LT lots for this scheme
                        scheme_lots = lots_df[
                            (lots_df['scheme'] == scheme) & 
                            (lots_df['is_lt'])
                        ].sort_values('purchase_date').copy()
                        
                        if len(scheme_lots) == 0:
                            continue
                        
                        current_nav = scheme_lots.iloc[0]['current_nav']
                        
                        # Calculate FIFO until ltcg_budget reached
                        target_gain = ltcg_budget
                        cumulative_gain = 0
                        cumulative_units = 0
                        cumulative_value = 0
                        
                        for _, lot in scheme_lots.iterrows():
                            if cumulative_gain >= target_gain:
                                break
                            
                            lot_gain = lot['gain_loss']
                            remaining_gain = target_gain - cumulative_gain
                            
                            if lot_gain <= remaining_gain:
                                cumulative_gain += lot_gain
                                cumulative_units += lot['units']
                                cumulative_value += lot['current_value']
                            else:
                                gain_per_unit = current_nav - lot['purchase_price']
                                if gain_per_unit > 0:
                                    units_needed = remaining_gain / gain_per_unit
                                    units_needed = min(units_needed, lot['units'])
                                    
                                    partial_value = units_needed * current_nav
                                    partial_gain = units_needed * gain_per_unit
                                    
                                    cumulative_gain += partial_gain
                                    cumulative_units += units_needed
                                    cumulative_value += partial_value
                                break
                        
                        cumulative_invested = cumulative_value - cumulative_gain
                        
                        summary_data.append({
                            'Scheme': scheme,
                            'Units to Redeem': cumulative_units,
                            'Redemption Value': cumulative_value,
                            'LTCG': cumulative_gain,
                            'Cost Basis': cumulative_invested
                        })
                    
                    # Create summary dataframe
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Format for display
                    display_df = summary_df.copy()
                    display_df['Scheme'] = display_df['Scheme'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                    display_df['Units'] = display_df['Units to Redeem'].apply(lambda x: f"{x:.2f}")
                    display_df['Redeem₹'] = display_df['Redemption Value'].apply(lambda x: format_indian_rupee_compact(x))
                    display_df['LTCG₹'] = display_df['LTCG'].apply(lambda x: format_indian_rupee_compact(x))
                    display_df['Cost₹'] = display_df['Cost Basis'].apply(lambda x: format_indian_rupee_compact(x))
                    
                    final_display = display_df[['Scheme', 'Units', 'Cost₹', 'Redeem₹', 'LTCG₹']]
                    
                    st.dataframe(
                        final_display,
                        hide_index=True,
                        height=400,
                        use_container_width=True,
                        column_config={
                            "Scheme": st.column_config.TextColumn("Scheme", width="large"),
                            "Units": st.column_config.TextColumn("Units", width="small"),
                            "Cost₹": st.column_config.TextColumn("Cost₹", width="medium"),
                            "Redeem₹": st.column_config.TextColumn("Redeem₹", width="medium"),
                            "LTCG₹": st.column_config.TextColumn("LTCG₹", width="medium"),
                        }
                    )
                    
                    st.info("💡 Select a specific scheme from the dropdown above to see detailed lot-level breakdown")
                
                else:
                    # Show detailed view for selected scheme
                    scheme = selected_harvest_scheme
                    
                    # Get LT lots for this scheme, sorted by purchase date (FIFO)
                    scheme_lots = lots_df[
                        (lots_df['scheme'] == scheme) & 
                        (lots_df['is_lt'])
                    ].sort_values('purchase_date').copy()
                    
                    if len(scheme_lots) == 0:
                        st.warning(f"No long-term holdings found for {scheme}")
                    else:
                        current_nav = scheme_lots.iloc[0]['current_nav']
                        
                        st.markdown(f"### {scheme}")
                        st.markdown(f"**Current NAV:** ₹{current_nav:.2f}")
                        st.markdown("---")
                        
                        # Calculate FIFO until ltcg_budget reached
                        target_gain = ltcg_budget
                        cumulative_gain = 0
                        cumulative_units = 0
                        cumulative_value = 0
                        lots_to_sell = []
                        
                        for _, lot in scheme_lots.iterrows():
                            if cumulative_gain >= target_gain:
                                break
                            
                            lot_gain = lot['gain_loss']
                            remaining_gain = target_gain - cumulative_gain
                            
                            if lot_gain <= remaining_gain:
                                # Take full lot
                                lots_to_sell.append({
                                    'Purchase Date': lot['purchase_date'].strftime('%d-%b-%Y'),
                                    'Units': lot['units'],
                                    'Purchase Price': lot['purchase_price'],
                                    'Current NAV': current_nav,
                                    'Invested': lot['invested'],
                                    'Value': lot['current_value'],
                                    'Gain': lot_gain,
                                    'Type': 'Full'
                                })
                                cumulative_gain += lot_gain
                                cumulative_units += lot['units']
                                cumulative_value += lot['current_value']
                            else:
                                # Partial lot
                                gain_per_unit = current_nav - lot['purchase_price']
                                if gain_per_unit > 0:
                                    units_needed = remaining_gain / gain_per_unit
                                    units_needed = min(units_needed, lot['units'])
                                    
                                    partial_value = units_needed * current_nav
                                    partial_invested = units_needed * lot['purchase_price']
                                    partial_gain = partial_value - partial_invested
                                    
                                    lots_to_sell.append({
                                        'Purchase Date': lot['purchase_date'].strftime('%d-%b-%Y'),
                                        'Units': units_needed,
                                        'Purchase Price': lot['purchase_price'],
                                        'Current NAV': current_nav,
                                        'Invested': partial_invested,
                                        'Value': partial_value,
                                        'Gain': partial_gain,
                                        'Type': 'Partial'
                                    })
                                    cumulative_gain += partial_gain
                                    cumulative_units += units_needed
                                    cumulative_value += partial_value
                                break
                        
                        # Calculate cumulative invested (cost basis)
                        cumulative_invested = sum(lot['Invested'] for lot in lots_to_sell)
                        
                        # Calculate XIRR for lots to sell
                        harvest_cashflows = []
                        for lot in lots_to_sell:
                            # Parse date back from string
                            lot_date = datetime.strptime(lot['Purchase Date'], '%d-%b-%Y')
                            harvest_cashflows.append((lot_date, -lot['Invested']))
                        harvest_cashflows.append((datetime.now(), cumulative_value))
                        harvest_xirr = calculate_xirr(harvest_cashflows) if len(harvest_cashflows) > 1 else 0
                        
                        # Display summary metrics
                        st.markdown("### 📊 Tax Harvesting Summary")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Units to Sell", f"{cumulative_units:.2f}")
                        with col2:
                            st.metric("Total Cost Basis", f"₹{format_indian_number(cumulative_invested)}", 
                                     help="Total amount invested in these units")
                        with col3:
                            st.metric("Redemption Amount", f"₹{format_indian_number(cumulative_value)}",
                                     help="Total value you'll receive on redemption")
                        with col4:
                            st.metric("Capital Gain (LTCG)", f"₹{format_indian_number(cumulative_gain)}",
                                     help="Long-term capital gain from this redemption")
                        with col5:
                            st.metric("XIRR", f"{harvest_xirr:.2f}%",
                                     help="Annualized return on these lots")
                        
                        st.markdown("---")
                        
                        if len(lots_to_sell) > 0:
                            st.subheader("📋 Lots to Sell (FIFO order)")
                            lots_sell_df = pd.DataFrame(lots_to_sell)
                            
                            # Add summary row
                            summary_row = pd.DataFrame([{
                                'Purchase Date': 'TOTAL',
                                'Units': cumulative_units,
                                'Purchase Price': 0,
                                'Current NAV': 0,
                                'Invested': cumulative_invested,
                                'Value': cumulative_value,
                                'Gain': cumulative_gain,
                                'Type': f'XIRR: {harvest_xirr:.2f}%'
                            }])
                            lots_sell_with_total = pd.concat([lots_sell_df, summary_row], ignore_index=True)
                            
                            st.dataframe(
                                lots_sell_with_total.style.format({
                                    'Units': '{:.2f}',
                                    'Invested': '₹{:,.2f}',
                                    'Value': '₹{:,.2f}',
                                    'Gain': '₹{:,.2f}'
                                })
                            )
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 4:
            # TAB 4: Multi-Fund Strategy
            st.header("🎯 Multi-Fund Balanced Tax Harvesting Strategy")
            
            # LTCG Budget Setting
            st.subheader("⚙️ LTCG Budget")
            col1, col2 = st.columns([3, 1])
            with col1:
                ltcg_budget_input = st.number_input(
                    "**LTCG Budget (₹)**",
                    min_value=5000,
                    max_value=500000,
                    value=st.session_state.ltcg_budget,
                    step=5000,
                    help="Total Long-Term Capital Gains budget for tax harvesting across all selected funds",
                    key="ltcg_input_multi"
                )
                # Update session state
                st.session_state.ltcg_budget = ltcg_budget_input
                ltcg_budget = ltcg_budget_input
            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
            
            st.info("💡 **What is LTCG Budget?** This is the maximum long-term profit you want to realize by selling units across all selected funds. The strategy will distribute this target profit among the funds based on your chosen distribution method.")
            
            st.markdown("---")
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Strategy Overview</h3>
                <p>Select multiple funds and choose a distribution strategy to harvest <b>₹{format_indian_number(ltcg_budget)}</b> LTCG 
                across selected funds.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Get all equity schemes with LT holdings
            equity_lt_schemes = lots_df[(lots_df['fund_type'] == 'equity') & (lots_df['is_lt'])].groupby('scheme').agg({
                'gain_loss': 'sum',
                'current_value': 'sum',
                'units': 'sum'
            }).reset_index()
            equity_lt_schemes = equity_lt_schemes.sort_values('gain_loss', ascending=False)
            
            st.subheader("Select Funds for Balanced Strategy")
            
            # Initialize session state for this multiselect
            if 'multi_fund_selection' not in st.session_state:
                st.session_state.multi_fund_selection = []
            
            # Multi-select for schemes
            selected_schemes = st.multiselect(
                "Choose funds to include:",
                options=equity_lt_schemes['scheme'].tolist(),
                default=st.session_state.multi_fund_selection if all(s in equity_lt_schemes['scheme'].tolist() for s in st.session_state.multi_fund_selection) else [],
                key='multi_fund_selection',
                help="Select 2 or more funds for balanced strategy"
            )
            
            if len(selected_schemes) > 0:
                st.markdown("---")
                
                # Strategy selection
                if 'distribution_strategy' not in st.session_state:
                    st.session_state.distribution_strategy = "Equal LTCG"
                
                strategy = st.radio(
                    "Distribution Strategy",
                    ["Equal LTCG", "Equal Redemption Value", "Proportional to Holdings", "Proportional to LT Gain"],
                    key='distribution_strategy',
                    horizontal=True,
                    help="""
                    • Equal LTCG: Same capital gain from each fund
                    • Equal Redemption Value: Same redemption amount from each fund
                    • Proportional to Holdings: Distribute based on current value
                    • Proportional to LT Gain: Distribute based on available LT gain
                    """
                )
                
                st.markdown("---")
                st.subheader("Strategy Results")
                
                # Get selected schemes data
                selected_schemes_data = equity_lt_schemes[equity_lt_schemes['scheme'].isin(selected_schemes)]
                num_funds = len(selected_schemes)
                
                # Calculate allocation per fund based on strategy
                allocations = {}
                
                if strategy == "Equal LTCG":
                    # Equal LTCG per fund with redistribution of shortfalls
                    base_allocation = ltcg_budget / num_funds
                    
                    # First pass: calculate what each fund can actually contribute
                    first_pass_results = {}
                    for scheme in selected_schemes:
                        scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                        max_available_gain = scheme_data['gain_loss']
                        allocated = min(base_allocation, max_available_gain)
                        first_pass_results[scheme] = {
                            'allocated': allocated,
                            'max_available': max_available_gain,
                            'can_take_more': max_available_gain > base_allocation
                        }
                    
                    # Calculate shortfall
                    total_allocated = sum(r['allocated'] for r in first_pass_results.values())
                    shortfall = ltcg_budget - total_allocated
                    
                    # Second pass: redistribute shortfall to funds that can take more
                    if shortfall > 0:
                        funds_with_capacity = [s for s in selected_schemes if first_pass_results[s]['can_take_more']]
                        if len(funds_with_capacity) > 0:
                            # Distribute shortfall proportionally among funds with capacity
                            for scheme in funds_with_capacity:
                                available_capacity = first_pass_results[scheme]['max_available'] - first_pass_results[scheme]['allocated']
                                total_capacity = sum(first_pass_results[s]['max_available'] - first_pass_results[s]['allocated'] for s in funds_with_capacity)
                                
                                if total_capacity > 0:
                                    additional = min(shortfall * (available_capacity / total_capacity), available_capacity)
                                    first_pass_results[scheme]['allocated'] += additional
                    
                    # Final allocations
                    for scheme in selected_schemes:
                        allocations[scheme] = first_pass_results[scheme]['allocated']
                    
                    total_final = sum(allocations.values())
                    st.info(f"**Strategy:** Equal LTCG distribution (₹{format_indian_number(base_allocation)} target per fund) → Total achieved: ₹{format_indian_number(total_final)}")
                
                elif strategy == "Equal Redemption Value":
                    # Equal redemption value with shortfall redistribution
                    total_value = selected_schemes_data['current_value'].sum()
                    total_gain = selected_schemes_data['gain_loss'].sum()
                    avg_gain_pct = total_gain / (total_value - total_gain) if (total_value - total_gain) > 0 else 0.3
                    target_redemption_per_fund = ltcg_budget / (avg_gain_pct * num_funds)
                    
                    # First pass
                    first_pass_results = {}
                    for scheme in selected_schemes:
                        scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                        scheme_gain = scheme_data['gain_loss']
                        scheme_value = scheme_data['current_value']
                        max_available = scheme_gain
                        scheme_gain_pct = scheme_gain / (scheme_value - scheme_gain) if (scheme_value - scheme_gain) > 0 else 0.3
                        target_ltcg = target_redemption_per_fund * scheme_gain_pct
                        allocated = min(target_ltcg, max_available)
                        first_pass_results[scheme] = {
                            'allocated': allocated,
                            'max_available': max_available,
                            'can_take_more': max_available > target_ltcg
                        }
                    
                    # Redistribute shortfall
                    total_allocated = sum(r['allocated'] for r in first_pass_results.values())
                    shortfall = ltcg_budget - total_allocated
                    if shortfall > 0:
                        funds_with_capacity = [s for s in selected_schemes if first_pass_results[s]['can_take_more']]
                        if len(funds_with_capacity) > 0:
                            for scheme in funds_with_capacity:
                                available_capacity = first_pass_results[scheme]['max_available'] - first_pass_results[scheme]['allocated']
                                total_capacity = sum(first_pass_results[s]['max_available'] - first_pass_results[s]['allocated'] for s in funds_with_capacity)
                                if total_capacity > 0:
                                    additional = min(shortfall * (available_capacity / total_capacity), available_capacity)
                                    first_pass_results[scheme]['allocated'] += additional
                    
                    for scheme in selected_schemes:
                        allocations[scheme] = first_pass_results[scheme]['allocated']
                    
                    total_final = sum(allocations.values())
                    st.info(f"**Strategy:** Equal redemption value (~₹{format_indian_number(target_redemption_per_fund)} per fund) → Total LTCG: ₹{format_indian_number(total_final)}")
                
                elif strategy == "Proportional to Holdings":
                    # Proportional to current value with shortfall redistribution
                    total_value = selected_schemes_data['current_value'].sum()
                    
                    # First pass
                    first_pass_results = {}
                    for scheme in selected_schemes:
                        scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                        scheme_value = scheme_data['current_value']
                        max_available = scheme_data['gain_loss']
                        weight = scheme_value / total_value
                        target_ltcg = ltcg_budget * weight
                        allocated = min(target_ltcg, max_available)
                        first_pass_results[scheme] = {
                            'allocated': allocated,
                            'max_available': max_available,
                            'can_take_more': max_available > target_ltcg
                        }
                    
                    # Redistribute shortfall
                    total_allocated = sum(r['allocated'] for r in first_pass_results.values())
                    shortfall = ltcg_budget - total_allocated
                    if shortfall > 0:
                        funds_with_capacity = [s for s in selected_schemes if first_pass_results[s]['can_take_more']]
                        if len(funds_with_capacity) > 0:
                            for scheme in funds_with_capacity:
                                available_capacity = first_pass_results[scheme]['max_available'] - first_pass_results[scheme]['allocated']
                                total_capacity = sum(first_pass_results[s]['max_available'] - first_pass_results[s]['allocated'] for s in funds_with_capacity)
                                if total_capacity > 0:
                                    additional = min(shortfall * (available_capacity / total_capacity), available_capacity)
                                    first_pass_results[scheme]['allocated'] += additional
                    
                    for scheme in selected_schemes:
                        allocations[scheme] = first_pass_results[scheme]['allocated']
                    
                    total_final = sum(allocations.values())
                    st.info(f"**Strategy:** Proportional to holdings → Total LTCG: ₹{format_indian_number(total_final)}")
                
                elif strategy == "Proportional to LT Gain":
                    # Proportional to available LT gain (no shortfall by design)
                    total_gain = selected_schemes_data['gain_loss'].sum()
                    
                    # Cap total allocation to what's available
                    actual_budget = min(ltcg_budget, total_gain)
                    
                    for scheme in selected_schemes:
                        scheme_gain = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]['gain_loss']
                        weight = scheme_gain / total_gain if total_gain > 0 else 1.0 / num_funds
                        allocations[scheme] = actual_budget * weight
                    
                    total_final = sum(allocations.values())
                    st.info(f"**Strategy:** Proportional to available LT gains → Total LTCG: ₹{format_indian_number(total_final)}")
                
                # Calculate for each selected scheme
                results = []
                
                for scheme in selected_schemes:
                    allocated_ltcg_per_fund = allocations[scheme]
                    
                    # Get LT lots for this scheme
                    scheme_lots = lots_df[
                        (lots_df['scheme'] == scheme) & 
                        (lots_df['is_lt'])
                    ].copy()
                    
                    if len(scheme_lots) == 0:
                        continue
                    
                    # Sort by purchase date (FIFO)
                    scheme_lots = scheme_lots.sort_values('purchase_date')
                    
                    current_nav = scheme_lots.iloc[0]['current_nav']
                    
                    # Calculate FIFO lots to sell
                    target_gain = allocated_ltcg_per_fund
                    cumulative_gain = 0
                    cumulative_units = 0
                    cumulative_value = 0
                    cumulative_invested = 0
                    lots_info = []
                    
                    for _, lot in scheme_lots.iterrows():
                        lot_units = lot['units']
                        lot_price = lot['purchase_price']
                        lot_value = lot_units * current_nav
                        lot_invested = lot_units * lot_price
                        lot_gain = lot_value - lot_invested
                        
                        if cumulative_gain >= target_gain:
                            break
                        
                        remaining_gain = target_gain - cumulative_gain
                        
                        if lot_gain <= remaining_gain:
                            # Take full lot
                            lots_info.append({
                                'Purchase Date': lot['purchase_date'].strftime('%d-%b-%Y'),
                                'Units': lot_units,
                                'Purchase Price': lot_price,
                                'Current NAV': current_nav,
                                'Invested': lot_invested,
                                'Value': lot_value,
                                'Gain': lot_gain,
                                'Type': 'Full'
                            })
                            cumulative_gain += lot_gain
                            cumulative_units += lot_units
                            cumulative_value += lot_value
                            cumulative_invested += lot_invested
                        else:
                            # Take partial lot
                            gain_per_unit = current_nav - lot_price
                            if gain_per_unit > 0:
                                units_needed = remaining_gain / gain_per_unit
                                units_needed = min(units_needed, lot_units)
                                
                                partial_value = units_needed * current_nav
                                partial_invested = units_needed * lot_price
                                partial_gain = partial_value - partial_invested
                                
                                lots_info.append({
                                    'Purchase Date': lot['purchase_date'].strftime('%d-%b-%Y'),
                                    'Units': units_needed,
                                    'Purchase Price': lot_price,
                                    'Current NAV': current_nav,
                                    'Invested': partial_invested,
                                    'Value': partial_value,
                                    'Gain': partial_gain,
                                    'Type': 'Partial'
                                })
                                cumulative_gain += partial_gain
                                cumulative_units += units_needed
                                cumulative_value += partial_value
                                cumulative_invested += partial_invested
                            
                            break
                    
                    # Calculate XIRR for this fund's lots to sell
                    fund_cashflows = []
                    for lot in lots_info:
                        lot_date = datetime.strptime(lot['Purchase Date'], '%d-%b-%Y')
                        fund_cashflows.append((lot_date, -lot['Invested']))
                    fund_cashflows.append((datetime.now(), cumulative_value))
                    fund_xirr = calculate_xirr(fund_cashflows) if len(fund_cashflows) > 1 else 0
                    
                    results.append({
                        'Scheme': scheme,
                        'Allocated LTCG': cumulative_gain,
                        'Units to Sell': cumulative_units,
                        'Cost Basis': cumulative_invested,
                        'Value to Redeem': cumulative_value,
                        'XIRR': fund_xirr,
                        'Lots': lots_info
                    })
                
                # Display results
                total_ltcg = sum(r['Allocated LTCG'] for r in results)
                total_cost_basis = sum(r['Cost Basis'] for r in results)
                total_redemption = sum(r['Value to Redeem'] for r in results)
                
                # Calculate overall XIRR across all selected funds
                overall_cashflows = []
                for result in results:
                    for lot in result['Lots']:
                        lot_date = datetime.strptime(lot['Purchase Date'], '%d-%b-%Y')
                        overall_cashflows.append((lot_date, -lot['Invested']))
                overall_cashflows.append((datetime.now(), total_redemption))
                overall_xirr = calculate_xirr(overall_cashflows) if len(overall_cashflows) > 1 else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Cost Basis", f"₹{format_indian_number(total_cost_basis)}")
                with col2:
                    st.metric("Total Redemption", f"₹{format_indian_number(total_redemption)}")
                with col3:
                    st.metric("Total LTCG", f"₹{format_indian_number(total_ltcg)}")
                with col4:
                    st.metric("Overall XIRR", f"{overall_xirr:.2f}%",
                             help="Combined annualized return across all selected funds")
                with col5:
                    st.metric("Number of Funds", f"{len(results)}")
                
                st.markdown("---")
                
                # Show per-fund details
                for result in results:
                    # Create expander header with key metrics
                    expander_header = f"**{result['Scheme'][:60]}{'...' if len(result['Scheme']) > 60 else ''}** | 💰 Redeem: {result['Units to Sell']:.2f} units (₹{format_indian_number(result['Value to Redeem'])}) → LTCG: ₹{format_indian_number(result['Allocated LTCG'])}"
                    
                    with st.expander(expander_header):
                        st.markdown("### 📊 Tax Harvesting Summary")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Units to Sell", f"{result['Units to Sell']:.2f}")
                        with col2:
                            st.metric("Cost Basis", f"₹{format_indian_number(result['Cost Basis'])}",
                                     help="Total amount invested in these units")
                        with col3:
                            st.metric("Redemption Amount", f"₹{format_indian_number(result['Value to Redeem'])}",
                                     help="Total value you'll receive on redemption")
                        with col4:
                            st.metric("Capital Gain (LTCG)", f"₹{format_indian_number(result['Allocated LTCG'])}",
                                     help="Long-term capital gain from this redemption")
                        with col5:
                            st.metric("XIRR", f"{result['XIRR']:.2f}%",
                                     help="Annualized return on these lots")
                        
                        st.markdown("---")
                        
                        st.subheader("📋 Lots to Sell (FIFO)")
                        if len(result['Lots']) > 0:
                            lots_display_df = pd.DataFrame(result['Lots'])
                            
                            # Add summary row
                            summary_row = pd.DataFrame([{
                                'Purchase Date': 'TOTAL',
                                'Units': result['Units to Sell'],
                                'Purchase Price': 0,
                                'Current NAV': 0,
                                'Invested': result['Cost Basis'],
                                'Value': result['Value to Redeem'],
                                'Gain': result['Allocated LTCG'],
                                'Type': f"XIRR: {result['XIRR']:.2f}%"
                            }])
                            lots_with_total = pd.concat([lots_display_df, summary_row], ignore_index=True)
                            
                            st.dataframe(
                                lots_with_total.style.format({
                                    'Units': '{:.2f}',
                                    'Invested': '₹{:,.2f}',
                                    'Value': '₹{:,.2f}',
                                    'Gain': '₹{:,.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                
                # Visualization
                st.markdown("---")
                st.subheader("Redemption Distribution")
                
                results_df = pd.DataFrame(results)
                # Truncate long scheme names for better display
                results_df['Short Name'] = results_df['Scheme'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
                
                # Add formatted hover text
                results_df['redemption_text'] = results_df['Value to Redeem'].apply(format_indian_currency)
                results_df['ltcg_text'] = results_df['Allocated LTCG'].apply(format_indian_currency)
                results_df['units_text'] = results_df['Units to Sell'].apply(lambda x: f"{x:.2f} units")
                
                # Create custom data array in the correct order for hover
                results_df['hover_data'] = results_df.apply(
                    lambda row: [row['units_text'], row['redemption_text'], row['ltcg_text']], axis=1
                )
                
                fig = px.bar(
                    results_df,
                    y='Short Name',
                    x='Value to Redeem',
                    color='Allocated LTCG',
                    title='Redemption Value by Fund',
                    labels={'Value to Redeem': 'Redemption (₹)', 'Allocated LTCG': 'LTCG (₹)', 'Short Name': 'Fund'},
                    orientation='h',
                    height=max(400, len(results_df) * 60),
                    custom_data=['units_text', 'redemption_text', 'ltcg_text']
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Units: %{customdata[0]}<br>Redemption: %{customdata[1]}<br>LTCG: %{customdata[2]}<extra></extra>'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("👆 Please select at least one fund to see the balanced strategy")
            
            # Show donation banner at bottom
            show_donation_banner()
