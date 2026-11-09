from pathlib import Path
import re
import sys
import pandas as pd


def standardize_state_nfhs(row: pd.Series) -> str:
    """Standardise state names in NFHS to match NAS naming conventions."""
    st = str(row["state"]).strip()
    dist = str(row["district"]).strip()

    # Map Dadra & Nagar Haveli and Daman & Diu to the NAS pair
    if st == "Dadra and Nagar Haveli & Daman and Diu":
        if "dadra" in dist.lower():
            return "Dadra & Nagar Haveli"
        else:
            return "Daman & Diu"

    # Standardise Delhi and Maharashtra
    if st == "NCT of Delhi":
        return "Delhi"
    if st == "Maharastra":
        return "Maharashtra"

    # Consistent Jammu & Kashmir and Ladakh
    if st in ["Jammu and Kashmir", "Jammu & Kashmir"]:
        return "Jammu & Kashmir"
    if st == "Ladakh":
        return "Ladakh"

    return st


def standardize_state_nas(row: pd.Series) -> str:
    """Standardise state names in NAS."""
    st = str(row["state"]).strip()
    if st in ["Jammu and Kashmir", "Jammu & Kashmir"]:
        return "Jammu & Kashmir"
    return st


def normalize_district_name(name: str) -> str:
    """Normalise district names: lowercase, strip punctuation and extra spaces, replace '&' with 'and'."""
    s = str(name).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_analysis_table(
    nas_path: str = "data/clean/nas_district_wide_v2.csv",
    nfhs_path: str = "data/clean/nfhs_selected.csv",
    manual_matches_path: str = "data/clean/manual_matches.csv",
    output_path: str = "data/clean/analysis_table.csv",
):
    print("=" * 90)
    print("BUILDING ANALYSIS TABLE: NAS 2021 + NFHS-5")
    print("=" * 90)

    # (1) Load datasets
    nas_file = Path(nas_path)
    nfhs_file = Path(nfhs_path)

    if not nas_file.exists():
        raise FileNotFoundError(f"NAS dataset not found: {nas_file}")
    if not nfhs_file.exists():
        raise FileNotFoundError(f"NFHS dataset not found: {nfhs_file}")

    nas_df = pd.read_csv(nas_file)
    nfhs_df = pd.read_csv(nfhs_file)

    print(f"Loaded NAS dataset : {nas_df.shape[0]} districts across {nas_df['state'].nunique()} states")
    print(f"Loaded NFHS dataset: {nfhs_df.shape[0]} districts across {nfhs_df['state'].nunique()} states\n")

    # Standardise state names and clean district names
    nas_df["state_std"] = nas_df.apply(standardize_state_nas, axis=1)
    nfhs_df["state_std"] = nfhs_df.apply(standardize_state_nfhs, axis=1)

    nas_df["district_clean"] = nas_df["district"].astype(str).str.strip()
    nfhs_df["district_clean"] = nfhs_df["district"].astype(str).str.strip()

    nas_df["dist_norm"] = nas_df["district"].apply(normalize_district_name)
    nfhs_df["dist_norm"] = nfhs_df["district"].apply(normalize_district_name)

    # (2) Step 2a: Exact matches already found by match_nas_nfhs.py
    exact_matches = pd.merge(
        nas_df[["state_std", "district_clean", "dist_norm"]],
        nfhs_df[["state_std", "district_clean", "dist_norm"]],
        on=["state_std", "dist_norm"],
        how="inner",
        suffixes=("_nas", "_nfhs"),
    )

    exact_pairs = pd.DataFrame(
        {
            "state": exact_matches["state_std"],
            "nas_district": exact_matches["district_clean_nas"],
            "nfhs_district": exact_matches["district_clean_nfhs"],
            "match_source": "exact",
        }
    )
    # Add normalized district keys for robust matching across whitespace differences
    exact_pairs["dist_norm_nas"] = exact_matches["dist_norm"]
    exact_pairs["dist_norm_nfhs"] = exact_matches["dist_norm"]
    print(f"Exact matches identified: {len(exact_pairs)} pairs")

    # Step 2b: Add manual matches from data/clean/manual_matches.csv
    manual_file = Path(manual_matches_path)
    if not manual_file.exists():
        print(f"\n[ERROR] Required manual matches file not found: '{manual_file}'")
        print("Please create 'data/clean/manual_matches.csv' with the manual match pairs to review.")
        print(f"Execution stopped without creating '{output_path}'.\n")
        return False, None, None

    manual_df = pd.read_csv(manual_file)
    col_map = {}
    for c in manual_df.columns:
        clow = c.strip().lower()
        if "state" in clow:
            col_map[c] = "state"
        elif "nas" in clow or clow == "district":
            col_map[c] = "nas_district"
        elif "nfhs" in clow or "candidate" in clow:
            col_map[c] = "nfhs_district"

    manual_df = manual_df.rename(columns=col_map)
    if not all(k in manual_df for k in ["state", "nas_district", "nfhs_district"]):
        raise ValueError(
            f"'{manual_file}' must contain columns for state, nas_district, and nfhs_district. Found: {manual_df.columns.tolist()}"
        )

    manual_df["state"] = manual_df["state"].astype(str).str.strip()
    manual_df["nas_district"] = manual_df["nas_district"].astype(str).str.strip()
    manual_df["nfhs_district"] = manual_df["nfhs_district"].astype(str).str.strip()
    manual_df["dist_norm_nas"] = manual_df["nas_district"].apply(normalize_district_name)
    manual_df["dist_norm_nfhs"] = manual_df["nfhs_district"].apply(normalize_district_name)
    manual_df["match_source"] = "manual"
    manual_pairs = manual_df[["state", "nas_district", "nfhs_district", "dist_norm_nas", "dist_norm_nfhs", "match_source"]].copy()
    print(f"Manual matches loaded   : {len(manual_pairs)} pairs from '{manual_file}'")

    # Combine exact matches and manual matches
    all_matches = pd.concat([exact_pairs, manual_pairs], ignore_index=True)
    all_matches = all_matches.drop_duplicates(subset=["state", "dist_norm_nas", "dist_norm_nfhs"]).reset_index(drop=True)
    print(f"Total candidate matches : {len(all_matches)} pairs\n")

    # (3) Run Conflict Check
    print("=" * 90)
    print("STEP 3: RUNNING CONFLICT CHECK")
    print("=" * 90)

    # Conflict Type A: Any NFHS district matched to more than one NAS district (e.g. sub-districts)
    nfhs_counts = (
        all_matches.groupby(["state", "dist_norm_nfhs"])["dist_norm_nas"]
        .nunique()
        .reset_index(name="nas_count")
    )
    conflicting_nfhs = nfhs_counts[nfhs_counts["nas_count"] > 1]

    # Conflict Type B: Any NAS district matched to more than one NFHS district
    nas_counts = (
        all_matches.groupby(["state", "dist_norm_nas"])["dist_norm_nfhs"]
        .nunique()
        .reset_index(name="nfhs_count")
    )
    conflicting_nas = nas_counts[nas_counts["nfhs_count"] > 1]

    has_conflict = len(conflicting_nfhs) > 0 or len(conflicting_nas) > 0

    if has_conflict:
        print("CONFLICT DETECTED! Stopping without writing output file.\n")

        if len(conflicting_nfhs) > 0:
            print(f"Conflict Type A: NFHS District matched to MULTIPLE NAS Districts ({len(conflicting_nfhs)} instances):")
            for _, r in conflicting_nfhs.iterrows():
                st = r["state"]
                norm_nfhs = r["dist_norm_nfhs"]
                pairs = all_matches[(all_matches["state"] == st) & (all_matches["dist_norm_nfhs"] == norm_nfhs)]
                nfhs_disp = pairs.iloc[0]["nfhs_district"]
                print(f"  * State: '{st}' | NFHS District: '{nfhs_disp}' matched to {r['nas_count']} NAS Districts:")
                for _, p in pairs.iterrows():
                    print(f"      - NAS: '{p['nas_district']}' (Source: {p['match_source']})")
            print()

        if len(conflicting_nas) > 0:
            print(f"Conflict Type B: NAS District matched to MULTIPLE NFHS Districts ({len(conflicting_nas)} instances):")
            for _, r in conflicting_nas.iterrows():
                st = r["state"]
                norm_nas = r["dist_norm_nas"]
                pairs = all_matches[(all_matches["state"] == st) & (all_matches["dist_norm_nas"] == norm_nas)]
                nas_disp = pairs.iloc[0]["nas_district"]
                print(f"  * State: '{st}' | NAS District: '{nas_disp}' matched to {r['nfhs_count']} NFHS Districts:")
                for _, p in pairs.iterrows():
                    print(f"      - NFHS: '{p['nfhs_district']}' (Source: {p['match_source']})")
            print()

        print("=" * 90)
        print("EXECUTION STOPPED: Please resolve the above conflicts in data/clean/manual_matches.csv.")
        print(f"The output file '{output_path}' was NOT created.")
        print("=" * 90)
        return False, all_matches, None

    print("No conflicts detected! All matches are clean 1-to-1 pairings.\n")

    # (4) Merge and save data/clean/analysis_table.csv
    print("=" * 90)
    print("STEP 4: MERGING AND SAVING ANALYSIS TABLE")
    print("=" * 90)

    # Exclude redundant join columns from NAS and NFHS to prevent _x, _y collisions
    nas_feature_cols = [
        c for c in nas_df.columns
        if c not in ["state", "district", "dist_norm", "state_std"]
    ]
    nfhs_feature_cols = [
        c for c in nfhs_df.columns
        if c not in ["state", "district", "dist_norm", "state_std"]
    ]

    # Merge NAS features on (state, dist_norm_nas)
    merged = pd.merge(
        all_matches,
        nas_df[["state_std", "dist_norm"] + [c for c in nas_feature_cols if c != "district_clean"]],
        left_on=["state", "dist_norm_nas"],
        right_on=["state_std", "dist_norm"],
        how="inner",
    ).drop(columns=["state_std", "dist_norm"])

    # Merge NFHS features on (state, dist_norm_nfhs)
    merged = pd.merge(
        merged,
        nfhs_df[["state_std", "dist_norm"] + [c for c in nfhs_feature_cols if c != "district_clean"]],
        left_on=["state", "dist_norm_nfhs"],
        right_on=["state_std", "dist_norm"],
        how="inner",
        suffixes=("_nas", "_nfhs"),
    ).drop(columns=["state_std", "dist_norm", "dist_norm_nas", "dist_norm_nfhs"])

    # Organize final column layout: identifiers first, then features
    id_cols = ["state", "nas_district", "nfhs_district", "match_source"]
    feature_cols = [c for c in merged.columns if c not in id_cols]
    final_analysis_df = merged[id_cols + feature_cols]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_analysis_df.to_csv(out_file, index=False)
    print(f"Saved analysis table to: {out_file}")
    print(f"Table shape: {final_analysis_df.shape} (rows, columns)\n")

    # (5) Print final number of rows, districts per state, and dropped NAS districts
    print("=" * 90)
    print("STEP 5: FINAL ANALYSIS TABLE REPORT")
    print("=" * 90)
    print(f"Total Rows in Analysis Table: {len(final_analysis_df)}")
    print(f"Total States Represented    : {final_analysis_df['state'].nunique()}\n")

    print("Number of Districts per State in Analysis Table:")
    state_counts = final_analysis_df["state"].value_counts().sort_index()
    for st, cnt in state_counts.items():
        print(f"  {st:<35}: {cnt:>2} districts")
    print()

    # Identify NAS districts that did not make it into the table with reasons
    matched_nas_set = set(zip(all_matches["state"], all_matches["nas_district"]))
    unmatched_nas = nas_df[~nas_df.apply(lambda r: (r["state_std"], r["district_clean"]) in matched_nas_set, axis=1)]

    # Check which states had zero NFHS candidates vs dropped during review
    nfhs_states_set = set(nfhs_df["state_std"])
    print(f"NAS Districts NOT in Analysis Table ({len(unmatched_nas)} districts):")

    for state, grp in unmatched_nas.groupby("state_std"):
        if state not in nfhs_states_set:
            reason = "No NFHS counterpart state"
        elif state == "Manipur":
            reason = "NFHS surveyed 9 districts (all exact matched); 7 newly created NAS districts have no NFHS counterpart"
        else:
            reason = "Unmatched during district reconciliation (needs manual review/decision)"

        print(f"\n  * {state} ({len(grp)} districts) - Reason: {reason}:")
        for dist in sorted(grp["district_clean"]):
            print(f"      - {dist}")

    print("=" * 90 + "\n")
    return True, all_matches, final_analysis_df


if __name__ == "__main__":
    build_analysis_table()
