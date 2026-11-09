from pathlib import Path
import re
import numpy as np
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize text by stripping, lowercasing, and collapsing multiple whitespaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def find_column_by_prefix(columns: list, target_prefix: str) -> str:
    """Match a column by starting text ignoring case, whitespace, and footnote digits."""
    norm_prefix = normalize_text(target_prefix)
    # Direct startswith
    for c in columns:
        if normalize_text(c).startswith(norm_prefix):
            return c
    # Strip non-alpha characters from the end of target prefix and try again
    clean_prefix = re.sub(r"[\d\(\)%]+$", "", norm_prefix).strip()
    for c in columns:
        if normalize_text(c).startswith(clean_prefix):
            return c
    # Fallback to token matching
    tokens = clean_prefix.split()[:4]
    for c in columns:
        norm_c = normalize_text(c)
        if all(t in norm_c for t in tokens):
            return c
    raise ValueError(f"Could not find matching column for prefix: '{target_prefix}'")


def create_nfhs_selected(
    input_path: str = "data/clean/nfhs_clean.csv",
    output_path: str = "data/clean/nfhs_selected.csv",
):
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading cleaned NFHS data from: {in_file}")
    df = pd.read_csv(in_file)
    print(f"Loaded shape: {df.shape}")

    # Column specifications: (short_name, target_prefix)
    column_specs = [
        ("state", "State/UT"),
        ("district", "District Names"),
        ("n_households", "Number of Households surveyed"),
        ("women_10yr", "Women (age 15-49) with 10 or more years of schooling"),
        ("women_literate", "Women (age 15-49) who are literate"),
        ("clean_fuel", "Households using clean fuel for cooking"),
        ("electricity", "Population living in households with electricity"),
        (
            "sanitation",
            "Population living in households that use an improved sanitation facility",
        ),
        ("early_marriage", "Women age 20-24 years married before age 18 years"),
        ("stunting", "Children under 5 years who are stunted"),
        (
            "pre_primary",
            "Children age 5 years who attended pre-primary school during the school year",
        ),
    ]

    # Non-flag columns
    candidate_cols = [c for c in df.columns if not c.endswith("_flag")]

    # Match each column
    matched_mapping = {}
    print("\nColumn Matching:")
    for short_name, prefix in column_specs:
        matched_col = find_column_by_prefix(candidate_cols, prefix)
        matched_mapping[short_name] = matched_col
        print(f"  {short_name:<16} <- '{matched_col}'")

    # Extract selected columns without dropping any rows
    df_selected = pd.DataFrame()
    for short_name, matched_col in matched_mapping.items():
        df_selected[short_name] = df[matched_col]

    # Calculate amenities_index as mean of z-scores of clean_fuel, electricity, sanitation
    # Using sample standard deviation (ddof=1)
    z_clean_fuel = (
        df_selected["clean_fuel"] - df_selected["clean_fuel"].mean()
    ) / df_selected["clean_fuel"].std()
    z_electricity = (
        df_selected["electricity"] - df_selected["electricity"].mean()
    ) / df_selected["electricity"].std()
    z_sanitation = (
        df_selected["sanitation"] - df_selected["sanitation"].mean()
    ) / df_selected["sanitation"].std()

    df_selected["amenities_index"] = (
        z_clean_fuel + z_electricity + z_sanitation
    ) / 3.0

    # Save to CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(out_file, index=False)
    print(f"\nSaved selected dataset to: {out_file}")

    # 1. Print Shape
    print("=" * 80)
    print("DATASET SHAPE")
    print("=" * 80)
    print(
        f"Rows: {df_selected.shape[0]}, Columns: {df_selected.shape[1]} (no rows dropped)\n"
    )

    # 2. Summary Statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    numeric_cols = [
        c
        for c in df_selected.columns
        if c not in ["state", "district"]
    ]
    summary_stats = df_selected[numeric_cols].describe().T
    summary_stats["missing"] = df_selected[numeric_cols].isna().sum()
    summary_stats["missing_pct"] = (
        df_selected[numeric_cols].isna().mean() * 100
    )
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_stats)
    print()

    # 3. Correlation Matrix for the 6 specified variables
    analysis_vars = [
        "women_10yr",
        "clean_fuel",
        "electricity",
        "sanitation",
        "early_marriage",
        "stunting",
    ]

    print("=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)
    corr_matrix = df_selected[analysis_vars].corr()
    print(corr_matrix.round(4))
    print()

    # 4. Variance Inflation Factor (VIF)
    print("=" * 80)
    print("VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 80)
    # VIF is diagonal of the inverse correlation matrix: VIF_j = 1 / (1 - R_j^2)
    inv_corr = np.linalg.inv(corr_matrix.values)
    vifs = np.diag(inv_corr)

    vif_df = pd.DataFrame(
        {
            "Variable": analysis_vars,
            "VIF": vifs,
            "Tolerance (1/VIF)": 1.0 / vifs,
            "R2_with_others": 1.0 - (1.0 / vifs),
        }
    )
    print(vif_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)

    return df_selected, summary_stats, corr_matrix, vif_df


if __name__ == "__main__":
    create_nfhs_selected()
