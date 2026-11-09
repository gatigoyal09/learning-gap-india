import os
from pathlib import Path
import pdfplumber

def explore_pdfs(target_dir: str = "data/raw/NAS_data21/Andaman_&_Nicobar_Islands"):
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"Directory not found: {target_path}")
        return

    # Walk directory and collect all PDF files
    pdf_files = []
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(Path(root) / file)

    pdf_files.sort()
    
    print(f"Directory: {target_path}")
    print(f"Found {len(pdf_files)} PDF file(s):")
    for idx, pdf in enumerate(pdf_files, start=1):
        print(f"  {idx}. {pdf.name} ({pdf})")
    
    if not pdf_files:
        print("No PDF files found.")
        return

    # Inspect the first PDF
    first_pdf = pdf_files[0]
    print("\n" + "=" * 80)
    print(f"Extracting raw text from first PDF: {first_pdf.name}")
    print("=" * 80 + "\n")

    with pdfplumber.open(first_pdf) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        for i, page in enumerate(pdf.pages, start=1):
            print(f"--- PAGE {i} ---")
            text = page.extract_text()
            if text:
                print(text)
            else:
                print("[No text found on this page]")
            print()

if __name__ == "__main__":
    explore_pdfs()
