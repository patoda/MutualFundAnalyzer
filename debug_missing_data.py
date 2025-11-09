import pdfplumber
import re
from datetime import datetime
import pandas as pd

pdf_path = r'c:\Users\anpatodi\Downloads\ACXXXXXX5J_01012010-09112025_CP198015134_09112025112206024.pdf'
password = 'Ankit@1993'

with pdfplumber.open(pdf_path, password=password) as pdf:
    # Check page 1 summary
    page1 = pdf.pages[0].extract_text()
    
    print("PORTFOLIO SUMMARY FROM PAGE 1:")
    print("=" * 80)
    
    for line in page1.split('\n'):
        match = re.search(r'(.+?(?:Mutual Fund|MF))\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)', line)
        if match:
            amc = match.group(1).strip()
            cost = float(match.group(2).replace(',', ''))
            market = float(match.group(3).replace(',', ''))
            print(f"{amc:50} Cost: ₹{cost:>12,.2f}  Market: ₹{market:>12,.2f}")
    
    # Look for total line
    for line in page1.split('\n'):
        if 'Total' in line or 'TOTAL' in line:
            print(f"\n{line}")
    
    print("\n" + "=" * 80)
    print("\nNow checking scheme parsing from transaction pages...")
    print("=" * 80)
    
    # Track all folios and schemes found
    folios_found = set()
    schemes_found = {}  # scheme -> list of folios
    
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        current_folio = None
        current_scheme = None
        
        for line in text.split('\n'):
            # Folio header
            folio_match = re.search(r'Folio No:\s*(\S+)', line)
            if folio_match:
                current_folio = folio_match.group(1)
                folios_found.add(current_folio)
                current_scheme = None
                continue
            
            # Scheme name - this is the pattern used in the app
            if current_folio and not current_scheme:
                scheme_match = re.search(r'^[A-Z0-9]+-(.+?)(?:\s*\(Non-Demat\)|\s*\(Demat\)|\s*-\s*Registrar|\s*-\s*ISIN)', line)
                if scheme_match:
                    current_scheme = scheme_match.group(1).strip()
                    if current_scheme not in schemes_found:
                        schemes_found[current_scheme] = []
                    schemes_found[current_scheme].append(current_folio)
                    continue
    
    print(f"\nTotal unique folios found: {len(folios_found)}")
    print(f"Total unique schemes found: {len(schemes_found)}")
    print(f"\nSchemes and their folios:")
    for scheme, folios in sorted(schemes_found.items()):
        print(f"  {scheme[:60]:60} -> {len(folios)} folio(s): {', '.join(set(folios))}")
    
    # Now check for "Unknown" schemes - these might be missing scheme names
    print("\n" + "=" * 80)
    print("Checking for transactions without proper scheme names...")
    print("=" * 80)
    
    unknown_count = 0
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        current_folio = None
        current_scheme = None
        
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
            
            # Transaction line without scheme name
            if current_folio and not current_scheme and re.match(r'\d{2}-[A-Za-z]{3}-\d{4}', line):
                if '***' not in line and 'To ' not in line:
                    unknown_count += 1
                    if unknown_count <= 10:
                        print(f"Page {page_num+1}, Folio {current_folio}: Transaction without scheme name")
                        print(f"  Line: {line[:100]}")
    
    print(f"\nTotal transactions without scheme names: {unknown_count}")
