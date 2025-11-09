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
    page_title="Portfolio Analyzer - Tax Harvesting & Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Mutual Fund Portfolio Analyzer with Tax Harvesting Strategies"
    }
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
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h3 style="margin: 0 0 1rem 0; color: white;">☕ Buy me a treat :)</h3>
            <div style="display: inline-block; background-color: rgba(0,0,0,0.2); 
                        padding: 0.5rem 1rem; border-radius: 0.3rem; font-family: monospace; 
                        font-size: 1.1rem; user-select: all; cursor: text;">
                {upi_id}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # UPI deep link button for mobile users
        st.markdown(f"""
        <div style="text-align: center; margin-top: 0.5rem;">
            <a href="upi://pay?pa={upi_id}&pn=Portfolio Analyzer&cu=INR" 
               style="display: inline-block; background-color: #4CAF50; color: white; 
                      padding: 0.8rem 2rem; text-decoration: none; border-radius: 0.5rem; 
                      font-weight: bold; font-size: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                📱 Pay via UPI App
            </a>
            <p style="font-size: 0.75rem; opacity: 0.7; margin-top: 0.5rem;">
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
            <p style="margin-top: 0.5rem; font-weight: bold; color: #667eea; font-size: 0.9rem;">Scan to Treat</p>
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
        return 'international', 1095  # 36 months = 3 years in days
    
    # Debt funds (36 months LTCG)
    debt_keywords = ['liquid', 'ultra short', 'low duration', 'money market', 'overnight',
                    'gilt', 'debt', 'bond', 'income', 'credit', 'banking & psu',
                    'corporate bond', 'dynamic bond', 'floating rate', 'short duration',
                    'medium duration', 'long duration']
    if any(keyword in scheme_lower for keyword in debt_keywords):
        return 'debt', 1095  # 36 months = 3 years in days
    
    # Default: Equity (12 months LTCG)
    return 'equity', 365  # 12 months = 1 year in days


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
                
                # Scheme name - more flexible pattern to catch various formats
                if current_folio and not current_scheme:
                    # Try primary pattern (with prefix like HGFGT-)
                    # Match until we see "Registrar" or "(Non-Demat)" or "ISIN"
                    scheme_match = re.search(r'^[A-Z0-9]+-(.+?)\s+(?:Registrar|ISIN)', line)
                    if not scheme_match:
                        # Try pattern with (Non-Demat) or (Demat)
                        scheme_match = re.search(r'^[A-Z0-9]+-(.+?)\s*\((?:Non-)?Demat\)', line)
                    
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
                
                # Skip stamp duty lines and header lines like "To 09-Nov-2025"
                if '***' in line or re.search(r'To \d{2}-[A-Za-z]{3}-\d{4}', line):
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


def calculate_realized_gains_current_fy(transactions_df):
    """Calculate realized capital gains (LTCG/STCG) for the current financial year using FIFO method."""
    
    # Determine current FY
    today = datetime.now()
    if today.month >= 4:  # April to December
        fy_start = datetime(today.year, 4, 1)
        fy_end = datetime(today.year + 1, 3, 31)
    else:  # January to March
        fy_start = datetime(today.year - 1, 4, 1)
        fy_end = datetime(today.year, 3, 31)
    
    # Filter redemption transactions in current FY (units < 0 means redemption)
    redemptions = transactions_df[
        (transactions_df['units'] < 0) &
        (transactions_df['date'] >= fy_start) &
        (transactions_df['date'] <= fy_end)
    ].copy()
    
    if len(redemptions) == 0:
        return None, []
    
    # Group by scheme and calculate gains
    schemes = redemptions['scheme'].unique()
    
    summary_data = []
    detailed_data = []
    
    for scheme in schemes:
        scheme_redemptions = redemptions[redemptions['scheme'] == scheme].sort_values('date')
        scheme_purchases = transactions_df[
            (transactions_df['scheme'] == scheme) &
            (transactions_df['units'] > 0) &  # units > 0 means purchase
            (transactions_df['date'] < fy_end)  # All purchases before FY end
        ].sort_values('date').copy()
        
        if len(scheme_purchases) == 0:
            continue
        
        # Get fund type using classify_fund_type function
        fund_type, ltcg_days = classify_fund_type(scheme)
        
        # Track remaining units in each purchase lot
        purchase_lots = []
        for _, purchase in scheme_purchases.iterrows():
            purchase_lots.append({
                'date': purchase['date'],
                'units': purchase['units'],
                'price': purchase['price'],
                'remaining_units': purchase['units']
            })
        
        scheme_ltcg = 0
        scheme_stcg = 0
        
        # Process each redemption
        for _, redemption in scheme_redemptions.iterrows():
            sell_date = redemption['date']
            sell_units = abs(redemption['units'])  # Make positive (units are negative for redemptions)
            sell_price = redemption['price']
            sell_value = sell_units * sell_price
            
            remaining_to_sell = sell_units
            matched_lots = []
            txn_ltcg = 0
            txn_stcg = 0
            
            # Match with purchase lots (FIFO)
            for lot in purchase_lots:
                if remaining_to_sell <= 0:
                    break
                
                if lot['remaining_units'] > 0:
                    # Calculate holding period
                    holding_days = (sell_date - lot['date']).days
                    is_lt = holding_days > ltcg_days  # Use ltcg_days from fund classification
                    
                    # Determine how many units to take from this lot
                    units_from_lot = min(remaining_to_sell, lot['remaining_units'])
                    
                    # Calculate gain from this lot
                    invested = units_from_lot * lot['price']
                    redemption_value = units_from_lot * sell_price
                    gain = redemption_value - invested
                    
                    # Classify as LT or ST
                    if is_lt:
                        txn_ltcg += gain
                    else:
                        txn_stcg += gain
                    
                    # Record matched lot
                    matched_lots.append({
                        'purchase_date': lot['date'],
                        'units': units_from_lot,
                        'purchase_price': lot['price'],
                        'invested': invested,
                        'redemption_value': redemption_value,
                        'gain': gain,
                        'holding_days': holding_days,
                        'is_lt': is_lt
                    })
                    
                    # Update remaining units
                    lot['remaining_units'] -= units_from_lot
                    remaining_to_sell -= units_from_lot
            
            # Store detailed transaction
            detailed_data.append({
                'scheme': scheme,
                'fund_type': fund_type,
                'sell_date': sell_date,
                'sell_units': sell_units,
                'sell_price': sell_price,
                'sell_value': sell_value,
                'ltcg': txn_ltcg,
                'stcg': txn_stcg,
                'matched_lots': matched_lots
            })
            
            scheme_ltcg += txn_ltcg
            scheme_stcg += txn_stcg
        
        # Store summary for scheme
        summary_data.append({
            'scheme': scheme,
            'fund_type': fund_type,
            'ltcg': scheme_ltcg,
            'stcg': scheme_stcg,
            'total_gain': scheme_ltcg + scheme_stcg,
            'num_transactions': len(scheme_redemptions)
        })
    
    summary_df = pd.DataFrame(summary_data) if summary_data else None
    
    return summary_df, detailed_data


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

# Enhanced Professional CSS
st.markdown("""
<style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 0 2rem;
    }
    
    /* Header styling with gradient text */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: rgba(31, 119, 180, 0.05);
        padding: 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid rgba(31, 119, 180, 0.8);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .success-card {
        background: rgba(40, 167, 69, 0.05);
        padding: 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid rgba(40, 167, 69, 0.8);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .success-card:hover {
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        transform: translateY(-2px);
    }
    
    .warning-card {
        background: rgba(255, 193, 7, 0.05);
        padding: 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid rgba(255, 193, 7, 0.8);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .warning-card:hover {
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        transform: translateY(-2px);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Compact but elegant metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.8rem !important;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
    }
    
    /* Professional dataframe styling */
    div[data-testid="stDataFrame"] {
        font-size: 0.85rem !important;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.85rem !important;
    }
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th {
        font-size: 0.85rem !important;
        padding: 0.5rem 0.75rem !important;
    }
    div[data-testid="stDataFrame"] th {
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px;
        background: rgba(102, 126, 234, 0.08) !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        font-size: 0.85rem !important;
        padding: 0.5rem 0.75rem !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(102, 126, 234, 0.05);
        padding: 0.5rem;
        border-radius: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 0.5rem;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 0.75rem;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Spinner styling */
    div[data-testid="stSpinner"] > div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
        border-bottom-color: #667eea !important;
        border-left-color: #764ba2 !important;
        border-width: 3px !important;
    }
    div[data-testid="stSpinner"] p {
        color: #667eea !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 0.75rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.02);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced App title with subtitle
st.markdown('''
<div style="text-align: center; padding: 0.5rem 0 0.5rem 0;">
    <div class="main-header">💼 Portfolio Analyzer</div>
    <p style="font-size: 1.1rem; opacity: 0.7; font-weight: 400; margin-top: -1rem;">
        Professional Mutual Fund Analytics & Tax Harvesting Platform
    </p>
</div>
''', unsafe_allow_html=True)

# Initialize session state
if 'show_landing' not in st.session_state:
    st.session_state.show_landing = True  # Start on landing page
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Portfolio Overview will be index 0 when radio buttons appear
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'ltcg_budget' not in st.session_state:
    st.session_state.ltcg_budget = 100000
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None

# Home button callback
def go_home():
    st.session_state.show_landing = True
    st.session_state.active_tab = 0

# Tab change callback
def on_tab_change():
    st.session_state.active_tab = st.session_state.nav_radio
    st.session_state.show_landing = False

# Home button at the top - always visible
if st.button("🏠 Home", key="home_btn", help="Return to landing page to upload new file"):
    go_home()
    st.rerun()

st.markdown("---")

# Tab names - no longer includes Landing Page
tab_names = ["📊 Portfolio Overview", "📋 All Holdings", "💼 Realized Gains", "💰 Single-Scheme Tax Harvest", "🎯 Multi-Fund Strategy"]

# Show radio buttons only when NOT on landing page AND data is processed
if not st.session_state.show_landing and st.session_state.processed_data is not None:
    # Ensure nav_radio matches active_tab when active_tab is changed programmatically
    if 'nav_radio' in st.session_state and st.session_state.nav_radio != st.session_state.active_tab:
        st.session_state.nav_radio = st.session_state.active_tab
    
    # Radio button for navigation
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
    
    active_tab = st.session_state.active_tab
else:
    # On landing page or no data - force landing view
    active_tab = None

# LANDING PAGE
if active_tab is None:
    # Upload section with better styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Step 1: Upload Your CAS PDF")
        uploaded_pdf = st.file_uploader(
            "Choose your Consolidated Account Statement (CAS) PDF file", 
            type=['pdf'], 
            help="Upload the CAS PDF file you received from CAMS/Karvy via email",
            label_visibility="collapsed"
        )
        
        # Track file changes - clear processed data ONLY when a different file is uploaded
        if uploaded_pdf is not None:
            current_file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
            
            # Check if this is a different file than before
            if 'last_uploaded_file_id' in st.session_state:
                if st.session_state.last_uploaded_file_id != current_file_id:
                    # Different file uploaded - clear old data AND cache
                    st.session_state.processed_data = None
                    st.cache_data.clear()  # Clear Streamlit's cache
                    st.session_state.last_uploaded_file_id = current_file_id
                    st.success(f"✅ File uploaded: **{uploaded_pdf.name}** ({uploaded_pdf.size / 1024:.1f} KB)")
            else:
                # First file upload
                st.session_state.last_uploaded_file_id = current_file_id
                st.success(f"✅ File uploaded: **{uploaded_pdf.name}** ({uploaded_pdf.size / 1024:.1f} KB)")
        
    with col2:
        st.markdown("### 🔐 Step 2: Enter Password")
        password = st.text_input(
            "PDF Password (if any)", 
            type="password", 
            help="Enter the password from your email. Leave blank if your PDF is not password protected",
            placeholder="Leave blank if no password",
            label_visibility="collapsed"
        )
    
    # Process button with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    process_clicked = False
    if uploaded_pdf:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            process_clicked = st.button(
                "🚀 Process & Analyze Portfolio", 
                type="primary", 
                use_container_width=True,
                help="Click to parse the CAS PDF and generate portfolio analytics"
            )
    
    # Process PDF when button is clicked
    if process_clicked and uploaded_pdf:
        # Clear cache before processing to ensure fresh data
        st.cache_data.clear()
        
        # Enhanced progress indicators
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.markdown("🔍 **Reading PDF file...**")
        progress_bar.progress(20)
        
        # Reset file pointer to beginning
        uploaded_pdf.seek(0)
        
        # Read file bytes
        pdf_bytes = uploaded_pdf.read()
        
        try:
            progress_text.markdown("📊 **Extracting transactions...**")
            progress_bar.progress(40)
            
            # Process PDF - use None if password is empty
            pdf_password = password if password else None
            
            progress_text.markdown("💼 **Calculating FIFO lots...**")
            progress_bar.progress(60)
            
            data = process_pdf(pdf_bytes, pdf_password)
            
            progress_text.markdown("✨ **Finalizing analytics...**")
            progress_bar.progress(80)
            
            # Check what we got
            if data is None:
                progress_bar.empty()
                progress_text.empty()
                st.error("❌ **📄 No valid CAS data found! This doesn't appear to be a valid Consolidated Account Statement. Please ensure you uploaded the correct PDF.**")
            elif 'lots_df' not in data or len(data['lots_df']) == 0:
                progress_bar.empty()
                progress_text.empty()
                st.error("❌ **📄 No holdings found in CAS! The PDF was processed but no current holdings were found. You may have zero balance in all schemes.**")
            elif 'transactions_df' not in data or len(data['transactions_df']) == 0:
                progress_bar.empty()
                progress_text.empty()
                st.error("❌ **📄 No transactions found in CAS! The PDF format might not be supported.**")
            else:
                progress_bar.progress(100)
                progress_text.markdown("✅ **Processing complete!**")
                time.sleep(0.5)
                
                # Store in session state
                st.session_state.processed_data = data
                
                # Show success with stats
                st.success(f"""
                ✅ **Portfolio Loaded Successfully!**
                - 📈 **{len(data['transactions_df'])}** transactions processed
                - 🏦 **{data['lots_df']['scheme'].nunique()}** unique schemes
                - 📦 **{len(data['lots_df'])}** FIFO lots tracked
                - 💰 **₹{data['lots_df']['current_value'].sum()/100000:.2f}L** total portfolio value
                """)
                
                # Switch to Portfolio Overview and hide landing page
                st.session_state.show_landing = False
                st.session_state.active_tab = 0  # Portfolio Overview is now index 0
                
                time.sleep(1)
                # Rerun to show radio buttons and portfolio
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
        st.success("✅ **Portfolio data loaded!**")
        
        data = st.session_state.processed_data
        st.info(f"📊 Loaded: {len(data['transactions_df'])} transactions | {data['lots_df']['scheme'].nunique()} schemes | {len(data['lots_df'])} lots")
        
        st.write("📌 **Use the navigation tabs above to explore your portfolio, or upload a new file to replace current data.**")
        
        # Show donation banner at bottom
        show_donation_banner()
    else:
        # Show welcome screen when no file uploaded or currently processing
        st.subheader("Welcome! 👋")
        
        st.write("This tool helps you analyze your mutual fund portfolio and optimize tax harvesting strategies.")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%); 
                    padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid rgba(102, 126, 234, 0.6); margin: 1rem 0;">
            <h4 style="margin: 0 0 1rem 0; color: rgba(102, 126, 234, 1);">📋 How to get your CAS (Consolidated Account Statement):</h4>
            <ol style="margin: 0; padding-left: 1.5rem; line-height: 1.8;">
                <li>Visit <a href="https://www.camsonline.com/Investors/Statements/Consolidated-Account-Statement" target="_blank" style="color: rgba(102, 126, 234, 1); text-decoration: none; font-weight: 500;">CAMS website</a> to download "CAS - CAMS+KFintech" statement</li>
                <li>For statement type, select <strong>"Detailed (Includes transaction listing)"</strong></li>
                <li>For Period, select <strong>"Specific Period"</strong></li>
                <li>In From Date, select a date far in the past since the inception of your investments</li>
                <li>In To Date, select latest date (today)</li>
                <li>In Folio Listing, select <strong>"Transacted folios and folios with balanceinfo"</strong></li>
                <li>Enter your email ID registered with mutual funds</li>
                <li>Choose a password for your CAS PDF</li>
                <li>Submit and check your email</li>
                <li>You'll receive the PDF with password in the email</li>
                <li>Upload that PDF here!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
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
elif active_tab is not None:
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
        
        st.markdown("---")
        
        # Display content based on selected tab
        if active_tab == 0:
            # Portfolio Overview
            st.header("Portfolio Overview")
            
            # Show quick info bar at top - only on Portfolio Overview tab
            st.info(f"📊 Loaded: {len(transactions_df)} transactions | {lots_df['scheme'].nunique()} schemes | {len(lots_df)} lots")
            
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
        
        elif active_tab == 1:
            # All Holdings
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
                # Keep numeric values for proper sorting
                final_df = display_df[['Scheme', 'Type', 'Total Units', 'LT Units', 'ST Units', 'Invested', 'Current Value', 'Gain/Loss', 'Gain %', 'XIRR']].copy()
                final_df.columns = ['Scheme', 'Type', 'Units', 'LT', 'ST', 'Invested₹', 'Value₹', 'Gain₹', 'Gain%', 'XIRR%']
                
                st.dataframe(
                    final_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                    column_config={
                        "Scheme": st.column_config.TextColumn("Scheme", width="large"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Units": st.column_config.NumberColumn("Units", format="%.0f", width="small"),
                        "LT": st.column_config.NumberColumn("LT", format="%.0f", width="small"),
                        "ST": st.column_config.NumberColumn("ST", format="%.0f", width="small"),
                        "Invested₹": st.column_config.NumberColumn("Invested₹", format="₹%.0f", width="medium"),
                        "Value₹": st.column_config.NumberColumn("Value₹", format="₹%.0f", width="medium"),
                        "Gain₹": st.column_config.NumberColumn("Gain₹", format="₹%.0f", width="medium"),
                        "Gain%": st.column_config.NumberColumn("Gain%", format="%.1f%%", width="small"),
                        "XIRR%": st.column_config.NumberColumn("XIRR%", format="%.1f%%", width="small"),
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
                            
                            # Show LT lots - keep numeric for sorting
                            lt_display = lt_lots.sort_values('purchase_date').copy()
                            lt_display_df = lt_display[['purchase_date', 'units', 'purchase_price', 'invested', 'current_value', 'gain_loss', 'gain_pct', 'cagr', 'holding_days']].copy()
                            lt_display_df['gain_pct'] = lt_display_df['gain_pct'] * 100  # Convert to percentage
                            lt_display_df.columns = ['Date', 'Units', 'Buy₹', 'Inv₹', 'Val₹', 'Gain₹', 'Gain%', 'CAGR%', 'Days']
                            
                            st.dataframe(
                                lt_display_df,
                                hide_index=True,
                                column_config={
                                    "Date": st.column_config.DateColumn("Date", format="DD-MMM-YY"),
                                    "Units": st.column_config.NumberColumn("Units", format="%.1f"),
                                    "Buy₹": st.column_config.NumberColumn("Buy₹", format="%.1f"),
                                    "Inv₹": st.column_config.NumberColumn("Inv₹", format="₹%.0f"),
                                    "Val₹": st.column_config.NumberColumn("Val₹", format="₹%.0f"),
                                    "Gain₹": st.column_config.NumberColumn("Gain₹", format="₹%.0f"),
                                    "Gain%": st.column_config.NumberColumn("Gain%", format="%.1f%%"),
                                    "CAGR%": st.column_config.NumberColumn("CAGR%", format="%.1f%%"),
                                    "Days": st.column_config.NumberColumn("Days", format="%.0f")
                                }
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
                            
                            # Show ST lots - keep numeric for sorting
                            st_display = st_lots.sort_values('purchase_date').copy()
                            st_display_df = st_display[['purchase_date', 'units', 'purchase_price', 'invested', 'current_value', 'gain_loss', 'gain_pct', 'cagr', 'holding_days']].copy()
                            st_display_df['gain_pct'] = st_display_df['gain_pct'] * 100  # Convert to percentage
                            st_display_df.columns = ['Date', 'Units', 'Buy₹', 'Inv₹', 'Val₹', 'Gain₹', 'Gain%', 'CAGR%', 'Days']
                            
                            st.dataframe(
                                st_display_df,
                                hide_index=True,
                                column_config={
                                    "Date": st.column_config.DateColumn("Date", format="DD-MMM-YY"),
                                    "Units": st.column_config.NumberColumn("Units", format="%.1f"),
                                    "Buy₹": st.column_config.NumberColumn("Buy₹", format="%.1f"),
                                    "Inv₹": st.column_config.NumberColumn("Inv₹", format="₹%.0f"),
                                    "Val₹": st.column_config.NumberColumn("Val₹", format="₹%.0f"),
                                    "Gain₹": st.column_config.NumberColumn("Gain₹", format="₹%.0f"),
                                    "Gain%": st.column_config.NumberColumn("Gain%", format="%.1f%%"),
                                    "CAGR%": st.column_config.NumberColumn("CAGR%", format="%.1f%%"),
                                    "Days": st.column_config.NumberColumn("Days", format="%.0f")
                                }
                            )
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 2:
            # Realized Gains (Current Financial Year)
            st.header("Realized Gains - Current Financial Year")
            
            # Determine current FY
            today = datetime.now()
            if today.month >= 4:  # April to December
                fy_start = datetime(today.year, 4, 1)
                fy_end = datetime(today.year + 1, 3, 31)
                fy_label = f"FY {today.year}-{str(today.year + 1)[-2:]}"
            else:  # January to March
                fy_start = datetime(today.year - 1, 4, 1)
                fy_end = datetime(today.year, 3, 31)
                fy_label = f"FY {today.year - 1}-{str(today.year)[-2:]}"
            
            st.info(f"ℹ️ Showing realized gains for **{fy_label}** ({fy_start.strftime('%d %b %Y')} to {fy_end.strftime('%d %b %Y')})")
            
            # Calculate realized gains
            realized_summary_df, realized_details_list = calculate_realized_gains_current_fy(transactions_df)
            
            if realized_summary_df is None or len(realized_summary_df) == 0:
                st.warning("🔍 No redemption transactions found in the current financial year.")
                st.info("💡 This tab shows capital gains from mutual fund redemptions (sell transactions) in the current FY using FIFO method.")
            else:
                # ========== SUMMARY TABLE ==========
                st.subheader("📊 Summary by Scheme")
                
                # Calculate totals
                total_ltcg = realized_summary_df['ltcg'].sum()
                total_stcg = realized_summary_df['stcg'].sum()
                total_gain = total_ltcg + total_stcg
                total_txns = realized_summary_df['num_transactions'].sum()
                
                # Show aggregate metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # LTCG metric with color coding
                ltcg_color = "green" if total_ltcg > 0 else ("red" if total_ltcg < 0 else "black")
                ltcg_display = f"₹{format_indian_number(total_ltcg)}" if total_ltcg >= 0 else f"-₹{format_indian_number(abs(total_ltcg))}"
                col1.markdown(f"<div style='font-size: 14px; color: rgba(49, 51, 63, 0.6);'>Total LTCG</div><div style='font-size: 36px; font-weight: 600; color: {ltcg_color};'>{ltcg_display}</div>", unsafe_allow_html=True)
                
                # STCG metric with color coding
                stcg_color = "green" if total_stcg > 0 else ("red" if total_stcg < 0 else "black")
                stcg_display = f"₹{format_indian_number(total_stcg)}" if total_stcg >= 0 else f"-₹{format_indian_number(abs(total_stcg))}"
                col2.markdown(f"<div style='font-size: 14px; color: rgba(49, 51, 63, 0.6);'>Total STCG</div><div style='font-size: 36px; font-weight: 600; color: {stcg_color};'>{stcg_display}</div>", unsafe_allow_html=True)
                
                # Total gain with color coding
                total_color = "green" if total_gain > 0 else ("red" if total_gain < 0 else "black")
                total_display = f"₹{format_indian_number(total_gain)}" if total_gain >= 0 else f"-₹{format_indian_number(abs(total_gain))}"
                col3.markdown(f"<div style='font-size: 14px; color: rgba(49, 51, 63, 0.6);'>Total Gains</div><div style='font-size: 36px; font-weight: 600; color: {total_color};'>{total_display}</div>", unsafe_allow_html=True)
                
                # Schemes count
                col4.markdown(f"<div style='font-size: 14px; color: rgba(49, 51, 63, 0.6);'>Schemes</div><div style='font-size: 36px; font-weight: 600; color: black;'>{len(realized_summary_df)}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Format summary table
                display_summary = realized_summary_df.copy()
                display_summary['Scheme'] = display_summary['scheme'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                display_summary['Type'] = display_summary['fund_type'].str.upper()
                
                # Format with color coding for gains/losses
                def format_gain_with_color(val):
                    if val >= 0:
                        return f'<span style="color: green;">₹{format_indian_number(val)}</span>'
                    else:
                        return f'<span style="color: red;">-₹{format_indian_number(abs(val))}</span>'
                
                display_summary['LTCG'] = display_summary['ltcg'].apply(lambda x: format_gain_with_color(x))
                display_summary['STCG'] = display_summary['stcg'].apply(lambda x: format_gain_with_color(x))
                display_summary['Total Gain'] = display_summary['total_gain'].apply(lambda x: format_gain_with_color(x))
                display_summary['Txns'] = display_summary['num_transactions'].astype(int)
                
                final_summary = display_summary[['Scheme', 'Type', 'LTCG', 'STCG', 'Total Gain', 'Txns']]
                
                # Display with HTML rendering for colors
                st.markdown(final_summary.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ========== DETAILED VIEW ==========
                st.subheader("🔎 Detailed Transaction View")
                
                # Filter dropdown - only show schemes with non-zero gains
                schemes_with_gains = realized_summary_df['scheme'].tolist()
                
                if 'realized_gains_scheme' not in st.session_state:
                    st.session_state.realized_gains_scheme = schemes_with_gains[0]
                
                selected_scheme = st.selectbox(
                    "Select Scheme",
                    schemes_with_gains,
                    index=schemes_with_gains.index(st.session_state.realized_gains_scheme) if st.session_state.realized_gains_scheme in schemes_with_gains else 0,
                    key='realized_gains_scheme',
                    help="View detailed breakdown of redemptions and matched purchase lots"
                )
                
                # Filter details for selected scheme
                scheme_details = [d for d in realized_details_list if d['scheme'] == selected_scheme]
                
                if len(scheme_details) == 0:
                    st.warning(f"No transactions found for {selected_scheme}")
                else:
                    # Calculate scheme totals
                    scheme_ltcg = sum(d['ltcg'] for d in scheme_details)
                    scheme_stcg = sum(d['stcg'] for d in scheme_details)
                    scheme_total = scheme_ltcg + scheme_stcg
                    fund_type = scheme_details[0]['fund_type'].upper()
                    
                    # Quick summary bar
                    col1, col2, col3, col4 = st.columns(4)
                    
                    ltcg_display = f"₹{format_indian_number(scheme_ltcg)}" if scheme_ltcg >= 0 else f"-₹{format_indian_number(abs(scheme_ltcg))}"
                    stcg_display = f"₹{format_indian_number(scheme_stcg)}" if scheme_stcg >= 0 else f"-₹{format_indian_number(abs(scheme_stcg))}"
                    total_display = f"₹{format_indian_number(scheme_total)}" if scheme_total >= 0 else f"-₹{format_indian_number(abs(scheme_total))}"
                    
                    col1.metric("LTCG", ltcg_display)
                    col2.metric("STCG", stcg_display)
                    col3.metric("Total", total_display)
                    col4.metric("Fund Type", fund_type)
                    
                    st.markdown("---")
                    
                    # Separate LT and ST transactions
                    lt_transactions = [d for d in scheme_details if d['ltcg'] != 0]
                    st_transactions = [d for d in scheme_details if d['stcg'] != 0]
                    
                    # ========== LONG TERM GAINS ==========
                    if len(lt_transactions) > 0:
                        st.markdown("#### 📗 Long Term Capital Gains (LTCG)")
                        
                        for idx, txn in enumerate(lt_transactions, 1):
                            ltcg_val = txn['ltcg']
                            ltcg_display = f"₹{format_indian_number(ltcg_val)}" if ltcg_val >= 0 else f"-₹{format_indian_number(abs(ltcg_val))}"
                            ltcg_color = "green" if ltcg_val >= 0 else "red"
                            
                            with st.expander(f"**Redemption #{idx}** — {txn['sell_date'].strftime('%d %b %Y')} — ₹{format_indian_number(txn['sell_value'])} ({txn['sell_units']:.2f} units @ ₹{txn['sell_price']:.2f})", expanded=(idx==1)):
                                st.markdown(f"**LTCG:** <span style='color: {ltcg_color}; font-weight: bold;'>{ltcg_display}</span>", unsafe_allow_html=True)
                                
                                # Show matched lots
                                st.markdown("**Matched Purchase Lots (FIFO):**")
                                
                                lt_lots = [lot for lot in txn['matched_lots'] if lot['is_lt']]
                                
                                if len(lt_lots) > 0:
                                    lot_data = []
                                    for lot in lt_lots:
                                        lot_data.append({
                                            'Purchase Date': lot['purchase_date'].strftime('%d %b %Y'),
                                            'Units': f"{lot['units']:.2f}",
                                            'Buy Price': f"₹{lot['purchase_price']:.2f}",
                                            'Sell Price': f"₹{txn['sell_price']:.2f}",
                                            'Invested': format_indian_rupee_compact(lot['invested']),
                                            'Redeemed': format_indian_rupee_compact(lot['redemption_value']),
                                            'Gain': format_indian_rupee_compact(lot['gain']),
                                            'Holding': f"{lot['holding_days']} days"
                                        })
                                    
                                    lot_df = pd.DataFrame(lot_data)
                                    st.dataframe(lot_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("#### 📗 Long Term Capital Gains (LTCG)")
                        st.info("No long-term redemptions for this scheme in current FY")
                    
                    st.markdown("---")
                    
                    # ========== SHORT TERM GAINS ==========
                    if len(st_transactions) > 0:
                        st.markdown("#### 📕 Short Term Capital Gains (STCG)")
                        
                        for idx, txn in enumerate(st_transactions, 1):
                            stcg_val = txn['stcg']
                            stcg_display = f"₹{format_indian_number(stcg_val)}" if stcg_val >= 0 else f"-₹{format_indian_number(abs(stcg_val))}"
                            stcg_color = "green" if stcg_val >= 0 else "red"
                            
                            with st.expander(f"**Redemption #{idx}** — {txn['sell_date'].strftime('%d %b %Y')} — ₹{format_indian_number(txn['sell_value'])} ({txn['sell_units']:.2f} units @ ₹{txn['sell_price']:.2f})", expanded=(idx==1)):
                                st.markdown(f"**STCG:** <span style='color: {stcg_color}; font-weight: bold;'>{stcg_display}</span>", unsafe_allow_html=True)
                                
                                # Show matched lots
                                st.markdown("**Matched Purchase Lots (FIFO):**")
                                
                                st_lots = [lot for lot in txn['matched_lots'] if not lot['is_lt']]
                                
                                if len(st_lots) > 0:
                                    lot_data = []
                                    for lot in st_lots:
                                        lot_data.append({
                                            'Purchase Date': lot['purchase_date'].strftime('%d %b %Y'),
                                            'Units': f"{lot['units']:.2f}",
                                            'Buy Price': f"₹{lot['purchase_price']:.2f}",
                                            'Sell Price': f"₹{txn['sell_price']:.2f}",
                                            'Invested': format_indian_rupee_compact(lot['invested']),
                                            'Redeemed': format_indian_rupee_compact(lot['redemption_value']),
                                            'Gain': format_indian_rupee_compact(lot['gain']),
                                            'Holding': f"{lot['holding_days']} days"
                                        })
                                    
                                    lot_df = pd.DataFrame(lot_data)
                                    st.dataframe(lot_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("#### 📕 Short Term Capital Gains (STCG)")
                        st.info("No short-term redemptions for this scheme in current FY")
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 3:
            # Single-Scheme Tax Harvest
            st.header("Single-Scheme Tax Harvesting")
            
            # LTCG Budget Setting - Compact
            col1, col2 = st.columns([2, 2])
            with col1:
                ltcg_budget_input = st.number_input(
                    "**LTCG Target (₹)** — Max long-term profit to harvest per scheme",
                    min_value=5000,
                    max_value=500000,
                    value=st.session_state.ltcg_budget,
                    step=5000,
                    key="ltcg_input_single"
                )
                st.session_state.ltcg_budget = ltcg_budget_input
                ltcg_budget = ltcg_budget_input
            
            st.markdown("---")
            
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
                    
                    # Create summary dataframe - keep numeric for sorting
                    summary_df = pd.DataFrame(summary_data)
                    summary_df['Scheme'] = summary_df['Scheme'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                    final_display = summary_df[['Scheme', 'Units to Redeem', 'Cost Basis', 'Redemption Value', 'LTCG']].copy()
                    final_display.columns = ['Scheme', 'Units', 'Cost₹', 'Redeem₹', 'LTCG₹']
                    
                    st.dataframe(
                        final_display,
                        hide_index=True,
                        height=400,
                        use_container_width=True,
                        column_config={
                            "Scheme": st.column_config.TextColumn("Scheme", width="large"),
                            "Units": st.column_config.NumberColumn("Units", format="%.2f", width="small"),
                            "Cost₹": st.column_config.NumberColumn("Cost₹", format="₹%.0f", width="medium"),
                            "Redeem₹": st.column_config.NumberColumn("Redeem₹", format="₹%.0f", width="medium"),
                            "LTCG₹": st.column_config.NumberColumn("LTCG₹", format="₹%.0f", width="medium"),
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
                            
                            # Style the TOTAL row - theme-aware color
                            def highlight_total(row):
                                if row['Purchase Date'] == 'TOTAL':
                                    return ['background-color: rgba(102, 126, 234, 0.15); font-weight: bold'] * len(row)
                                return [''] * len(row)
                            
                            st.dataframe(
                                lots_sell_with_total.style.apply(highlight_total, axis=1),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Purchase Date": st.column_config.TextColumn("Purchase Date"),
                                    "Units": st.column_config.NumberColumn("Units", format="%.2f"),
                                    "Purchase Price": st.column_config.NumberColumn("Purchase Price", format="%.2f"),
                                    "Current NAV": st.column_config.NumberColumn("Current NAV", format="%.2f"),
                                    "Invested": st.column_config.NumberColumn("Invested", format="₹%.0f"),
                                    "Value": st.column_config.NumberColumn("Value", format="₹%.0f"),
                                    "Gain": st.column_config.NumberColumn("Gain", format="₹%.0f"),
                                    "Type": st.column_config.TextColumn("Type")
                                }
                            )
            
            # Show donation banner at bottom
            show_donation_banner()
        
        elif active_tab == 4:
            # Multi-Fund Strategy
            st.header("🎯 Multi-Fund Balanced Tax Harvesting Strategy")
            
            # LTCG Budget Setting - Compact
            col1, col2 = st.columns([2, 2])
            with col1:
                ltcg_budget_input = st.number_input(
                    "**LTCG Target (₹)** — Total profit to harvest across all selected funds",
                    min_value=5000,
                    max_value=500000,
                    value=st.session_state.ltcg_budget,
                    step=5000,
                    key="ltcg_input_multi"
                )
                st.session_state.ltcg_budget = ltcg_budget_input
                ltcg_budget = ltcg_budget_input
            
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
                
                st.info("ℹ️ All strategies aim to get as close as possible to your LTCG target without exceeding it.")
                
                st.subheader("Strategy Results")
                
                # Get selected schemes data
                selected_schemes_data = equity_lt_schemes[equity_lt_schemes['scheme'].isin(selected_schemes)]
                num_funds = len(selected_schemes)
                
                # Calculate allocation per fund based on strategy
                allocations = {}
                
                if strategy == "Equal LTCG":
                    # Multi-stage Equal LTCG: iteratively handle bottlenecks
                    # Stage 1: Try equal distribution
                    # Stage 2+: Lock bottleneck funds, recalculate equal for remaining funds
                    
                    remaining_budget = ltcg_budget
                    remaining_funds = set(selected_schemes)
                    locked_allocations = {}
                    
                    while remaining_funds and remaining_budget > 0:
                        target_per_fund = remaining_budget / len(remaining_funds)
                        bottlenecks = []
                        
                        for scheme in remaining_funds:
                            scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                            max_available = scheme_data['gain_loss']
                            
                            if max_available < target_per_fund:
                                # This fund is a bottleneck - lock it at max
                                locked_allocations[scheme] = max_available
                                remaining_budget -= max_available
                                bottlenecks.append(scheme)
                        
                        if not bottlenecks:
                            # No bottlenecks - allocate target to all remaining funds
                            for scheme in remaining_funds:
                                locked_allocations[scheme] = target_per_fund
                            break
                        else:
                            # Remove bottlenecks and recalculate for remaining
                            remaining_funds -= set(bottlenecks)
                    
                    allocations = locked_allocations
                    total_final = sum(allocations.values())
                    base_allocation = ltcg_budget / num_funds
                    st.info(f"**Strategy:** Equal LTCG (₹{format_indian_number(base_allocation)} target/fund, adjusted for bottlenecks) → Total: ₹{format_indian_number(total_final)}")
                
                elif strategy == "Equal Redemption Value":
                    # Multi-stage Equal Redemption: iteratively handle bottlenecks
                    # Calculate equal redemption value that yields LTCG within budget
                    
                    # Helper function to calculate LTCG from a redemption value for a scheme
                    def calc_ltcg_from_redemption(scheme, redemption_value):
                        scheme_lots = lots_df[(lots_df['scheme'] == scheme) & (lots_df['is_lt'])].copy()
                        if len(scheme_lots) == 0:
                            return 0, 0
                        
                        scheme_lots = scheme_lots.sort_values('purchase_date')
                        current_nav = scheme_lots.iloc[0]['current_nav']
                        
                        cumulative_gain = 0
                        cumulative_value = 0
                        
                        for _, lot in scheme_lots.iterrows():
                            if cumulative_value >= redemption_value:
                                break
                            lot_value = lot['units'] * current_nav
                            remaining_value = redemption_value - cumulative_value
                            
                            if lot_value <= remaining_value:
                                cumulative_gain += lot['gain_loss']
                                cumulative_value += lot_value
                            else:
                                # Partial lot
                                units_needed = remaining_value / current_nav
                                partial_invested = units_needed * lot['purchase_price']
                                partial_gain = remaining_value - partial_invested
                                cumulative_gain += partial_gain
                                cumulative_value += remaining_value
                                break
                        
                        max_available = scheme_lots['gain_loss'].sum()
                        return cumulative_gain, max_available
                    
                    # Estimate initial redemption value per fund
                    total_value = selected_schemes_data['current_value'].sum()
                    total_invested = total_value - selected_schemes_data['gain_loss'].sum()
                    overall_gain_ratio = selected_schemes_data['gain_loss'].sum() / total_invested if total_invested > 0 else 0
                    estimated_redemption_per_fund = ltcg_budget / (num_funds * overall_gain_ratio) if overall_gain_ratio > 0 else 0
                    
                    # Multi-stage: lock bottlenecks, recalculate equal redemption for remaining
                    remaining_budget = ltcg_budget
                    remaining_funds = set(selected_schemes)
                    locked_allocations = {}
                    
                    iteration = 0
                    max_iterations = num_funds  # Prevent infinite loops
                    
                    while remaining_funds and remaining_budget > 0 and iteration < max_iterations:
                        iteration += 1
                        
                        # Calculate target redemption for remaining funds
                        if iteration == 1:
                            target_redemption = estimated_redemption_per_fund
                        else:
                            # Recalculate based on remaining budget and funds
                            # Use average gain ratio of remaining funds
                            remaining_schemes_data = selected_schemes_data[selected_schemes_data['scheme'].isin(remaining_funds)]
                            remaining_total_value = remaining_schemes_data['current_value'].sum()
                            remaining_total_invested = remaining_total_value - remaining_schemes_data['gain_loss'].sum()
                            remaining_gain_ratio = remaining_schemes_data['gain_loss'].sum() / remaining_total_invested if remaining_total_invested > 0 else 0
                            
                            if remaining_gain_ratio > 0:
                                target_redemption = remaining_budget / (len(remaining_funds) * remaining_gain_ratio)
                            else:
                                target_redemption = remaining_budget / len(remaining_funds)
                        
                        bottlenecks = []
                        
                        for scheme in list(remaining_funds):
                            ltcg_from_target, max_available = calc_ltcg_from_redemption(scheme, target_redemption)
                            
                            if ltcg_from_target >= max_available * 0.99:  # Within 1% of max = bottleneck
                                # Lock at maximum
                                locked_allocations[scheme] = max_available
                                remaining_budget -= max_available
                                bottlenecks.append(scheme)
                            elif iteration == max_iterations - 1:
                                # Last iteration - allocate whatever we calculated
                                locked_allocations[scheme] = min(ltcg_from_target, remaining_budget)
                                remaining_budget -= locked_allocations[scheme]
                        
                        if not bottlenecks or iteration == max_iterations - 1:
                            # No bottlenecks - allocate to all remaining funds
                            for scheme in remaining_funds:
                                if scheme not in locked_allocations:
                                    ltcg_from_target, _ = calc_ltcg_from_redemption(scheme, target_redemption)
                                    locked_allocations[scheme] = min(ltcg_from_target, remaining_budget / len([s for s in remaining_funds if s not in locked_allocations]))
                            break
                        else:
                            # Remove bottlenecks and continue
                            remaining_funds -= set(bottlenecks)
                    
                    allocations = locked_allocations
                    total_final = sum(allocations.values())
                    st.info(f"**Strategy:** Equal redemption value (~₹{format_indian_number(estimated_redemption_per_fund)}/fund, {iteration} iterations) → Total LTCG: ₹{format_indian_number(total_final)}")
                
                elif strategy == "Proportional to Holdings":
                    # Multi-stage Proportional to Holdings: handle bottlenecks iteratively
                    
                    remaining_budget = ltcg_budget
                    remaining_funds = set(selected_schemes)
                    locked_allocations = {}
                    
                    while remaining_funds and remaining_budget > 0:
                        # Calculate proportions based on remaining funds only
                        remaining_schemes_data = selected_schemes_data[selected_schemes_data['scheme'].isin(remaining_funds)]
                        total_value = remaining_schemes_data['current_value'].sum()
                        
                        bottlenecks = []
                        
                        for scheme in remaining_funds:
                            scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                            scheme_value = scheme_data['current_value']
                            max_available = scheme_data['gain_loss']
                            
                            weight = scheme_value / total_value if total_value > 0 else 1.0 / len(remaining_funds)
                            target_ltcg = remaining_budget * weight
                            
                            if target_ltcg >= max_available:
                                # Bottleneck - lock at max
                                locked_allocations[scheme] = max_available
                                remaining_budget -= max_available
                                bottlenecks.append(scheme)
                        
                        if not bottlenecks:
                            # No bottlenecks - allocate proportionally to all remaining
                            for scheme in remaining_funds:
                                scheme_data = selected_schemes_data[selected_schemes_data['scheme'] == scheme].iloc[0]
                                scheme_value = scheme_data['current_value']
                                weight = scheme_value / total_value if total_value > 0 else 1.0 / len(remaining_funds)
                                locked_allocations[scheme] = remaining_budget * weight
                            break
                        else:
                            # Remove bottlenecks and recalculate
                            remaining_funds -= set(bottlenecks)
                    
                    allocations = locked_allocations
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
                
                # SAFETY CHECK: Ensure total LTCG doesn't exceed budget
                total_allocated = sum(allocations.values())
                if total_allocated > ltcg_budget:
                    # Scale down proportionally to fit within budget
                    scale_factor = ltcg_budget / total_allocated
                    for scheme in allocations:
                        allocations[scheme] *= scale_factor
                    st.warning(f"⚠️ Allocations scaled down to fit within budget of ₹{format_indian_number(ltcg_budget)}")
                
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
                            
                            # Style the TOTAL row - theme-aware color
                            def highlight_total_multi(row):
                                if row['Purchase Date'] == 'TOTAL':
                                    return ['background-color: rgba(102, 126, 234, 0.15); font-weight: bold'] * len(row)
                                return [''] * len(row)
                            
                            st.dataframe(
                                lots_with_total.style.apply(highlight_total_multi, axis=1),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Purchase Date": st.column_config.TextColumn("Purchase Date"),
                                    "Units": st.column_config.NumberColumn("Units", format="%.2f"),
                                    "Purchase Price": st.column_config.NumberColumn("Purchase Price", format="%.2f"),
                                    "Current NAV": st.column_config.NumberColumn("Current NAV", format="%.2f"),
                                    "Invested": st.column_config.NumberColumn("Invested", format="₹%.0f"),
                                    "Value": st.column_config.NumberColumn("Value", format="₹%.0f"),
                                    "Gain": st.column_config.NumberColumn("Gain", format="₹%.0f"),
                                    "Type": st.column_config.TextColumn("Type")
                                }
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
