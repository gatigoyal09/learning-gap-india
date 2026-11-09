from pathlib import Path
import pandas as pd


def explore_nfhs(data_dir: str = "data/raw/NFHS"):
    dir_path = Path(data_dir)
    if not dir_path.exists():
        # Fallback for lowercase/case-sensitivity
        dir_path = Path("data/raw/nfhs")
        
    if not dir_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    # Check for excel files first, then csv
    excel_files = list(dir_path.glob("*.xlsx")) + list(dir_path.glob("*.xls"))
    csv_files = list(dir_path.glob("*.csv"))

    sheets_data = {}

    if excel_files:
        file_path = excel_files[0]
        print(f"Reading Excel file: {file_path}")
        excel_obj = pd.ExcelFile(file_path)
        sheet_names = excel_obj.sheet_names
        print(f"Found {len(sheet_names)} sheet(s): {sheet_names}\n")
        for s in sheet_names:
            sheets_data[s] = pd.read_excel(excel_obj, sheet_name=s)
    elif csv_files:
        file_path = csv_files[0]
        sheet_name = file_path.name
        print(f"Reading file in '{dir_path}': {file_path}")
        print(f"Sheet name (single table): ['{sheet_name}']\n")
        sheets_data[sheet_name] = pd.read_csv(file_path)
    else:
        print(f"No Excel or CSV data file found in '{dir_path}'.")
        return

    for sheet_name, df in sheets_data.items():
        print("=" * 80)
        print(f"SHEET: {sheet_name}")
        print("=" * 80)
        print(f"Shape: {df.shape} (rows, columns)\n")

        print("First 10 Column Names:")
        for idx, col in enumerate(df.columns[:10], start=1):
            print(f"  {idx:>2}. {col}")
        print()

        print("First 5 Rows:")
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 1000)
        print(df.iloc[:5, :10])
        print("  ... (showing first 10 columns of first 5 rows)")
        print()

        # Identify state and district columns
        state_cols = [c for c in df.columns if any(k in c.lower() for k in ["state", "ut"])]
        dist_cols = [c for c in df.columns if "district" in c.lower()]

        print("=" * 80)
        print("STATE & DISTRICT ORIENTATION")
        print("=" * 80)
        if state_cols and dist_cols:
            state_col = state_cols[0]
            dist_col = dist_cols[0]
            num_states = df[state_col].nunique()
            num_districts_unique = df[dist_col].nunique()
            total_dist_rows = len(df)
            print(f"State column identified    : '{state_col}'")
            print(f"District column identified : '{dist_col}'")
            print(f"Unique States / UTs        : {num_states}")
            print(f"Unique District names      : {num_districts_unique}")
            print(f"Total District rows        : {total_dist_rows} (unique state-district pairs)")
            print("Districts in rows or cols? : Districts are in ROWS (each row represents a district,")
            print("                             and columns represent survey indicators).")
        else:
            print("Could not automatically identify state/district columns by name.")
        print()

        # Count cells with parentheses and cells with *
        print("=" * 80)
        print("SPECIAL CHARACTER CELL COUNTS")
        print("=" * 80)
        paren_count = 0
        star_count = 0

        for col in df.columns:
            s = df[col].astype(str)
            paren_count += s.str.contains(r"\(|\)", regex=True).sum()
            star_count += s.str.contains(r"\*", regex=True).sum()

        print(f"Cells containing parentheses '(' or ')' : {paren_count}")
        print(f"Cells containing '*'                     : {star_count}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    explore_nfhs()
