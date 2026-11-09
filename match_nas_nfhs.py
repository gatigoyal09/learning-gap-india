from difflib import SequenceMatcher
from pathlib import Path
import re
import pandas as pd


def standardize_state_nfhs(row: pd.Series) -> str:
    """Standardise state names in NFHS to match NAS naming conventions."""
    st = str(row["State/UT"]).strip()
    dist = str(row["District Names"]).strip()

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

    # Ensure consistent Jammu & Kashmir and Ladakh
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
    # Replace '&' with ' and '
    s = s.replace("&", " and ")
    # Replace all punctuation characters with a space
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse multiple whitespace characters into single space and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_similarity(s1: str, s2: str) -> float:
    """Compute fuzzy match similarity score considering both sequence and token sort."""
    ratio_seq = SequenceMatcher(None, s1, s2).ratio()
    tokens1 = " ".join(sorted(s1.split()))
    tokens2 = " ".join(sorted(s2.split()))
    ratio_tok = SequenceMatcher(None, tokens1, tokens2).ratio()
    return max(ratio_seq, ratio_tok)


def match_nas_nfhs(
    nas_path: str = "data/clean/nas_district_wide_v2.csv",
    nfhs_path: str = "data/clean/nfhs_clean.csv",
    review_output_path: str = "data/clean/match_review.csv",
):
    print("=" * 90)
    print("MATCHING NAS 2021 AND NFHS-5 DISTRICTS")
    print("=" * 90)

    # Load files
    nas_file = Path(nas_path)
    nfhs_file = Path(nfhs_path)
    if not nas_file.exists():
        raise FileNotFoundError(f"NAS file not found: {nas_file}")
    if not nfhs_file.exists():
        raise FileNotFoundError(f"NFHS file not found: {nfhs_file}")

    nas_df = pd.read_csv(nas_file)[["state", "district"]].copy()
    nfhs_df = pd.read_csv(nfhs_file)[["State/UT", "District Names"]].copy()

    print(f"Loaded NAS dataset : {nas_df.shape[0]} districts across {nas_df['state'].nunique()} states")
    print(f"Loaded NFHS dataset: {nfhs_df.shape[0]} districts across {nfhs_df['State/UT'].nunique()} states\n")

    # Step 1: Standardise state names
    nas_df["state_std"] = nas_df.apply(standardize_state_nas, axis=1)
    nfhs_df["state_std"] = nfhs_df.apply(standardize_state_nfhs, axis=1)

    # Step 2: Normalise district names
    nas_df["dist_norm"] = nas_df["district"].apply(normalize_district_name)
    nfhs_df["dist_norm"] = nfhs_df["District Names"].apply(normalize_district_name)

    # Clean district names (stripped of leading/trailing whitespace)
    nas_df["district_clean"] = nas_df["district"].astype(str).str.strip()
    nfhs_df["district_clean"] = nfhs_df["District Names"].astype(str).str.strip()

    # Step 3: Exact match on (state, district)
    exact_matches = pd.merge(
        nas_df[["state_std", "district_clean", "dist_norm"]],
        nfhs_df[["state_std", "district_clean", "dist_norm"]],
        on=["state_std", "dist_norm"],
        how="inner",
        suffixes=("_nas", "_nfhs"),
    )

    n_exact = len(exact_matches)
    matched_nas_keys = set(zip(exact_matches["state_std"], exact_matches["district_clean_nas"]))
    matched_nfhs_keys = set(zip(exact_matches["state_std"], exact_matches["district_clean_nfhs"]))

    # Unmatched subsets
    unmatched_nas = nas_df[
        ~nas_df.apply(lambda r: (r["state_std"], r["district_clean"]) in matched_nas_keys, axis=1)
    ].copy()

    unmatched_nfhs = nfhs_df[
        ~nfhs_df.apply(lambda r: (r["state_std"], r["district_clean"]) in matched_nfhs_keys, axis=1)
    ].copy()

    print(f"Step 3 Result: Exact Matches = {n_exact} districts")
    print(f"  Unmatched NAS districts : {len(unmatched_nas)}")
    print(f"  Unmatched NFHS districts: {len(unmatched_nfhs)}\n")

    # Step 4: Fuzzy match within the same state only (DO NOT auto-accept)
    review_records = []
    nas_no_candidate = {}
    assigned_nfhs_keys = set()

    # Process NAS unmatched rows
    for state, state_nas in unmatched_nas.groupby("state_std"):
        state_nfhs = unmatched_nfhs[unmatched_nfhs["state_std"] == state]

        if len(state_nfhs) == 0:
            # All unmatched NAS districts in this state have NO candidates in NFHS
            nas_no_candidate[state] = state_nas["district_clean"].tolist()
            continue

        # Evaluate candidate pairs within the same state
        for _, r_nas in state_nas.iterrows():
            scores = []
            for _, r_nfhs in state_nfhs.iterrows():
                sc = compute_similarity(r_nas["dist_norm"], r_nfhs["dist_norm"])
                scores.append((r_nfhs["district_clean"], sc))

            # Sort descending by score
            scores.sort(key=lambda x: x[1], reverse=True)
            best_cand, best_score = scores[0]
            assigned_nfhs_keys.add((state, best_cand))

            # Also capture runner-up if available
            second_cand = scores[1][0] if len(scores) > 1 else ""
            second_score = scores[1][1] if len(scores) > 1 else None

            review_records.append(
                {
                    "state": state,
                    "nas_district": r_nas["district_clean"],
                    "candidate_nfhs": best_cand,
                    "similarity_score": round(best_score, 4),
                    "runner_up_nfhs": second_cand,
                    "runner_up_score": round(second_score, 4) if second_score is not None else None,
                }
            )

    # Identify NFHS districts with no candidate at all:
    # 1. NFHS districts in states where NAS has NO unmatched districts
    # 2. NFHS districts that were not the best candidate for any NAS district in that state
    nfhs_no_candidate = {}
    for state, state_nfhs in unmatched_nfhs.groupby("state_std"):
        state_nas = unmatched_nas[unmatched_nas["state_std"] == state]
        if len(state_nas) == 0:
            nfhs_no_candidate[state] = state_nfhs["district_clean"].tolist()
        else:
            unassigned_in_state = [
                d for d in state_nfhs["district_clean"] if (state, d) not in assigned_nfhs_keys
            ]
            if unassigned_in_state:
                nfhs_no_candidate[state] = unassigned_in_state

    # Save candidates with scores to match_review.csv
    review_df = pd.DataFrame(review_records)
    review_df = review_df.sort_values(
        by=["state", "similarity_score"], ascending=[True, False]
    ).reset_index(drop=True)

    out_file = Path(review_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(out_file, index=False)
    print(f"Step 4 Result: Wrote {len(review_df)} review candidate pairs to: {out_file}\n")

    # Step 5: Summary Report
    print("=" * 90)
    print("MATCHING SUMMARY REPORT")
    print("=" * 90)
    print(f"1. Exact Matches                : {n_exact} districts")
    print(f"2. Rows Needing Manual Review   : {len(review_df)} districts (written to match_review.csv)")
    print(f"   (Fuzzy candidate pairs are NOT auto-accepted)")

    total_nas_no_cand = sum(len(dists) for dists in nas_no_candidate.values())
    print(f"\n3. NAS Districts with No Candidate at All ({total_nas_no_cand} districts):")
    if nas_no_candidate:
        for st, dists in sorted(nas_no_candidate.items()):
            print(f"   * {st} ({len(dists)}): {dists}")
    else:
        print("   None")

    total_nfhs_no_cand = sum(len(dists) for dists in nfhs_no_candidate.values())
    print(f"\n4. NFHS Districts with No Candidate at All ({total_nfhs_no_cand} districts):")
    if nfhs_no_candidate:
        for st, dists in sorted(nfhs_no_candidate.items()):
            print(f"   * {st} ({len(dists)}): {dists}")
    else:
        print("   None")
    print("=" * 90 + "\n")

    return exact_matches, review_df, nas_no_candidate, nfhs_no_candidate


if __name__ == "__main__":
    match_nas_nfhs()
