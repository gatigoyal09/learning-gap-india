from pathlib import Path
import numpy as np
import pandas as pd


def clean_nfhs(
    raw_path: str = "data/raw/NFHS/datafile.csv",
    output_path: str = "data/clean/nfhs_clean.csv",
):
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raw_file = Path("data/raw/nfhs/datafile.csv")

    if not raw_file.exists():
        raise FileNotFoundError(f"NFHS raw data file not found at {raw_path}")

    print(f"Loading raw NFHS data from: {raw_file}")
    df_raw = pd.read_csv(raw_file)
    print(f"Raw data shape: {df_raw.shape} (rows, columns)\n")

    id_cols = ["District Names", "State/UT"]
    indicator_cols = [c for c in df_raw.columns if c not in id_cols]

    # Build clean dataframe dictionary to avoid DataFrame fragmentation
    clean_dict = {}
    for col in id_cols:
        clean_dict[col] = df_raw[col]

    missing_records = []

    for col in indicator_cols:
        raw_series = df_raw[col].astype(str).str.strip()

        # Flag True where original value was enclosed in parentheses
        is_paren = raw_series.str.startswith("(") & raw_series.str.endswith(")")

        # Convert to numeric: strip parentheses, replace '*' with NaN
        cleaned_series = (
            raw_series.str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.strip()
        )
        cleaned_series = cleaned_series.replace("*", np.nan).replace(
            ["nan", "NaN", ""], np.nan
        )
        num_series = pd.to_numeric(cleaned_series, errors="coerce")

        clean_dict[col] = num_series
        clean_dict[f"{col}_flag"] = is_paren

        n_missing = num_series.isna().sum()
        pct_missing = (n_missing / len(df_raw)) * 100
        missing_records.append(
            {
                "indicator": col,
                "missing_count": n_missing,
                "missing_pct": pct_missing,
            }
        )

    # Assemble cleaned DataFrame
    df_clean = pd.DataFrame(clean_dict)

    # Ensure output directory exists and save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)
    print(f"Saved clean NFHS dataset to: {out_file}")
    print(f"Clean data shape: {df_clean.shape} (rows, columns)")
    print(f"Identifier columns: {len(id_cols)}")
    print(f"Numeric indicator columns: {len(indicator_cols)}")
    print(f"Companion flag columns: {len(indicator_cols)}\n")

    # Missing values report
    missing_df = pd.DataFrame(missing_records)
    missing_df = missing_df.sort_values(
        by=["missing_count", "indicator"], ascending=[False, True]
    ).reset_index(drop=True)

    print("=" * 110)
    print(f"{'RANK':<5} {'INDICATOR COLUMN':<85} {'MISSING':<10} {'PERCENT':<8}")
    print("=" * 110)
    for idx, row in missing_df.iterrows():
        print(
            f"{idx + 1:<5} {row['indicator']:<85} {row['missing_count']:<10} {row['missing_pct']:>6.2f}%"
        )
    print("=" * 110)

    # Summary metrics
    total_cells = len(df_raw) * len(indicator_cols)
    total_missing = missing_df["missing_count"].sum()
    pct_total_missing = (total_missing / total_cells) * 100
    print(f"\nTotal indicator cells: {total_cells:,}")
    print(f"Total missing values : {total_missing:,} ({pct_total_missing:.2f}%)")
    indicators_with_missing = (missing_df["missing_count"] > 0).sum()
    print(
        f"Indicators with at least one missing value: {indicators_with_missing} / {len(indicator_cols)}"
    )
    indicators_complete = (missing_df["missing_count"] == 0).sum()
    print(
        f"Indicators with zero missing values      : {indicators_complete} / {len(indicator_cols)}"
    )

    return df_clean, missing_df


if __name__ == "__main__":
    clean_nfhs()
