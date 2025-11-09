import sys
import os
sys.path.insert(0, r'c:\w\CG')

# We need streamlit for the app imports
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Import directly from portfolio_app
from portfolio_app import parse_pdf_cas, calculate_fifo_lots
import tempfile

pdf_path_original = r'c:\Users\anpatodi\Downloads\ACXXXXXX5J_01012010-09112025_CP198015134_09112025112206024.pdf'

# Read PDF bytes
with open(pdf_path_original, 'rb') as f:
    pdf_bytes = f.read()

# Write to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(pdf_bytes)
    pdf_path = tmp_file.name

try:
    # Parse PDF
    print("Parsing PDF...")
    transactions_df, current_navs = parse_pdf_cas(pdf_path, 'Ankit@1993')
    print(f"Transactions parsed: {len(transactions_df)}")
    
    # Calculate FIFO lots
    print("\nCalculating FIFO lots...")
    lots_df = calculate_fifo_lots(transactions_df)
    print(f"Lots calculated: {len(lots_df)}")
    
    # Calculate totals
    total_invested = lots_df['invested'].sum()
    print(f"\n{'='*60}")
    print(f"TOTAL INVESTED (from lots_df): ₹{total_invested:,.2f}")
    print(f"EXPECTED FROM PDF: ₹4,786,359.53")
    print(f"DIFFERENCE: ₹{total_invested - 4786359.53:,.2f}")
    print(f"{'='*60}")
    
    # Show breakdown by scheme
    print(f"\nBreakdown by scheme:")
    scheme_totals = lots_df.groupby('scheme')['invested'].sum().sort_values(ascending=False)
    for scheme, invested in scheme_totals.head(20).items():
        print(f"{scheme[:50]:50} ₹{invested:>12,.2f}")
    
    # Check if there are duplicate lots
    print(f"\nChecking for duplicates...")
    duplicates = lots_df.groupby(['scheme', 'folio', 'purchase_date', 'units']).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        print(f"WARNING: Found {len(duplicates)} potential duplicate lots!")
        print(duplicates)
    else:
        print("No obvious duplicates found")
    
finally:
    # Clean up
    if os.path.exists(pdf_path):
        os.unlink(pdf_path)
