import os
from pathlib import Path

def check_extraction(base_dir: str = "data/raw/NAS_data21"):
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory '{base_path}' does not exist.")
        return

    # Identify all state folders (immediate subdirectories)
    state_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not state_folders:
        print(f"No state folders found in '{base_path}'.")
        return

    results = []
    zero_pdf_states = []
    zip_containing_states = []

    total_pdfs = 0

    for state_folder in state_folders:
        state_name = state_folder.name
        pdf_count = 0
        zip_files = []

        for root, _, files in os.walk(state_folder):
            for file in files:
                file_lower = file.lower()
                if file_lower.endswith(".pdf"):
                    pdf_count += 1
                elif file_lower.endswith(".zip"):
                    zip_files.append(os.path.join(root, file))

        total_pdfs += pdf_count
        results.append((state_name, pdf_count))

        if pdf_count == 0:
            zero_pdf_states.append(state_name)

        if zip_files:
            zip_containing_states.append((state_name, zip_files))

    # Print table
    state_col_width = max(len(name) for name, _ in results)
    state_col_width = max(state_col_width, len("State / UT"))
    count_col_width = max(len(str(total_pdfs)), len("PDF Count"))

    header = f"{'State / UT':<{state_col_width}} | {'PDF Count':>{count_col_width}}"
    separator = f"{'-' * state_col_width}-+-{'-' * count_col_width}"

    print(header)
    print(separator)
    for state_name, count in results:
        print(f"{state_name:<{state_col_width}} | {count:>{count_col_width}}")
    print(separator)
    print(f"{'Total':<{state_col_width}} | {total_pdfs:>{count_col_width}}")
    print()

    # List state folders with zero PDFs
    print("State folders with 0 PDFs:")
    if zero_pdf_states:
        for s in zero_pdf_states:
            print(f"  - {s}")
    else:
        print("  None (all state folders contain at least one PDF)")
    print()

    # List state folders still containing .zip files
    print("State folders still containing .zip files:")
    if zip_containing_states:
        for s, zips in zip_containing_states:
            print(f"  - {s} ({len(zips)} .zip file(s)):")
            for z in zips:
                print(f"      {z}")
    else:
        print("  None (no .zip files found in any state folder)")

if __name__ == "__main__":
    check_extraction()
