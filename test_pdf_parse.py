import pdfplumber
import pandas as pd
import re
from datetime import datetime

pdf_path = r'c:\Users\anpatodi\Downloads\ACXXXXXX5J_01012010-09112025_CP198015134_09112025112206024.pdf'
password = 'Ankit@1993'

with pdfplumber.open(pdf_path, password=password) as pdf:
    # Extract all transactions
    transactions = []
    current_folio = None
    current_scheme = None
    
    for page_num, page in enumerate(pdf.pages):
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
            else:
                units_val, units_neg = all_numbers[-2]
                balance_val, _ = all_numbers[-1]
                
                units = -units_val if units_neg else units_val
                balance = balance_val
                amount = 0
                price = 0
            
            transactions.append({
                'date': txn_date,
                'scheme': current_scheme or 'Unknown',
                'folio': current_folio or 'Unknown',
                'units': units,
                'price': price,
                'amount': amount,
                'balance': balance
            })

# Create DataFrame
df = pd.DataFrame(transactions)

print(f"Total transactions: {len(df)}")
print(f"Unique schemes: {df['scheme'].nunique()}")
print(f"Unique folios: {df['folio'].nunique()}")
print(f"\nScheme-Folio combinations: {df.groupby(['scheme', 'folio']).size().shape[0]}")

# Check for duplicates
print(f"\n--- Checking for potential issues ---")

# Group by scheme and folio, check final balances
for (scheme, folio), group in df.groupby(['scheme', 'folio']):
    group_sorted = group.sort_values('date')
    final_balance = group_sorted.iloc[-1]['balance']
    
    if final_balance > 0.01:
        # Calculate FIFO
        purchase_queue = []
        for _, txn in group_sorted.iterrows():
            if txn['units'] > 0:
                if txn['amount'] == 0:
                    continue
                purchase_queue.append({
                    'units': txn['units'],
                    'amount': txn['amount']
                })
            elif txn['units'] < 0:
                units_to_redeem = abs(txn['units'])
                while units_to_redeem > 0.001 and purchase_queue:
                    lot = purchase_queue[0]
                    if lot['units'] <= units_to_redeem:
                        units_to_redeem -= lot['units']
                        purchase_queue.pop(0)
                    else:
                        lot['units'] -= units_to_redeem
                        units_to_redeem = 0
        
        total_invested = sum(lot['amount'] for lot in purchase_queue)
        total_units = sum(lot['units'] for lot in purchase_queue)
        
        print(f"\n{scheme[:40]:40} | Folio: {folio:15} | Units: {total_units:10.2f} | Invested: ₹{total_invested:,.2f}")

# Calculate totals
all_lots_invested = 0
for (scheme, folio), group in df.groupby(['scheme', 'folio']):
    group_sorted = group.sort_values('date')
    final_balance = group_sorted.iloc[-1]['balance']
    
    if final_balance > 0.01:
        purchase_queue = []
        for _, txn in group_sorted.iterrows():
            if txn['units'] > 0:
                if txn['amount'] == 0:
                    continue
                purchase_queue.append({
                    'units': txn['units'],
                    'amount': txn['amount']
                })
            elif txn['units'] < 0:
                units_to_redeem = abs(txn['units'])
                while units_to_redeem > 0.001 and purchase_queue:
                    lot = purchase_queue[0]
                    if lot['units'] <= units_to_redeem:
                        units_to_redeem -= lot['units']
                        purchase_queue.pop(0)
                    else:
                        lot['units'] -= units_to_redeem
                        units_to_redeem = 0
        
        all_lots_invested += sum(lot['amount'] for lot in purchase_queue)

print(f"\n{'='*80}")
print(f"CALCULATED TOTAL INVESTED: ₹{all_lots_invested:,.2f}")
print(f"EXPECTED FROM PDF: ₹4,786,359.53")
print(f"DIFFERENCE: ₹{all_lots_invested - 4786359.53:,.2f}")
