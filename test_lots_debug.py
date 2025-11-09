import sys
sys.path.insert(0, r'c:\w\CG')

# Import the actual functions from the app
import pdfplumber
import pandas as pd
import re
from datetime import datetime
import tempfile
import os

# Copy the exact parsing functions from portfolio_app.py
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
                
                if '***' in line or 'To 08-Nov-2025' in line or 'To 09-Nov-2025' in line:
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
    
    transactions_df = pd.DataFrame(transactions)
    return transactions_df, current_navs


pdf_path = r'c:\Users\anpatodi\Downloads\ACXXXXXX5J_01012010-09112025_CP198015134_09112025112206024.pdf'
transactions_df, current_navs = parse_pdf_cas(pdf_path, 'Ankit@1993')

print(f"Total transactions parsed: {len(transactions_df)}")
print(f"Date range: {transactions_df['date'].min()} to {transactions_df['date'].max()}")
print(f"\nSample transactions:")
print(transactions_df[['date', 'scheme', 'folio', 'units', 'amount', 'balance']].head(10))

# Check for switch transactions
switch_txns = transactions_df[transactions_df['description'].str.contains('switch', case=False, na=False)]
print(f"\nSwitch transactions: {len(switch_txns)}")
if len(switch_txns) > 0:
    print(switch_txns[['date', 'scheme', 'description', 'units', 'amount']].head(20))
