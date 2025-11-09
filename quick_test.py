import sys
sys.path.insert(0, r'c:\w\CG')

from portfolio_app import parse_pdf_cas, calculate_fifo_lots
import tempfile
import os

pdf_path = r'c:\Users\anpatodi\Downloads\ACXXXXXX5J_01012010-09112025_CP198015134_09112025112206024.pdf'

with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
tmp.write(pdf_bytes)
tmp.close()

try:
    print("Parsing PDF...")
    transactions_df, navs = parse_pdf_cas(tmp.name, 'Ankit@1993')
    print(f"Transactions: {len(transactions_df)}")
    
    print("\nCalculating FIFO lots...")
    lots_df = calculate_fifo_lots(transactions_df)
    print(f"Lots: {len(lots_df)}")
    
    total_invested = lots_df['invested'].sum()
    print(f"\nTotal Invested: Rs {total_invested:,.2f}")
    print(f"Expected: Rs 4,786,359.53")
    print(f"Difference: Rs {total_invested - 4786359.53:,.2f}")
    
    if abs(total_invested - 4786359.53) < 100000:
        print("\n✓ LOOKS GOOD!")
    else:
        print("\n✗ Still has issues")
        
finally:
    os.unlink(tmp.name)
