import pandas as pd
import numpy as np

# Load dataset
csv_path = "data/clean/nas_district_scores.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print(f"LOADED DATASET: {csv_path}")
print(f"Shape: {df.shape} (rows, columns)")
print("=" * 80)

# (1) Number of districts per state
print("\n" + "=" * 80)
print("(1) NUMBER OF DISTRICTS PER STATE")
print("=" * 80)
districts_per_state = df.groupby("state")["district"].nunique().sort_values(ascending=False)
for state, count in districts_per_state.items():
    print(f"  {state:<30} : {count:>3} districts")
print("-" * 80)
print(f"  TOTAL STATES/UTs               : {len(districts_per_state):>3}")
print(f"  TOTAL UNIQUE DISTRICTS (state-district pairs): {df[['state', 'district']].drop_duplicates().shape[0]:>3}")

# (2) Any district_score below 0 or above 100
print("\n" + "=" * 80)
print("(2) DISTRICT_SCORE OUT OF RANGE (< 0 OR > 100)")
print("=" * 80)
# Ensure numeric
score_numeric = pd.to_numeric(df["district_score"], errors="coerce")
out_of_range = df[score_numeric.notna() & ((score_numeric < 0) | (score_numeric > 100))]
if out_of_range.empty:
    print("  None. All non-missing district_score values are within [0, 100].")
    print(f"  Min score: {score_numeric.min()}, Max score: {score_numeric.max()}")
else:
    print(f"  Found {len(out_of_range)} row(s) with district_score out of range:")
    print(out_of_range[["state", "district", "class", "subject", "district_score"]])

# (3) Any missing values per column
print("\n" + "=" * 80)
print("(3) MISSING VALUES PER COLUMN")
print("=" * 80)
missing_counts = df.isna().sum()
empty_str_counts = (df == "").sum()
total_missing = missing_counts + empty_str_counts
for col in df.columns:
    print(f"  {col:<20} : {total_missing[col]:>5} missing ({missing_counts[col]} NaN, {empty_str_counts[col]} empty string)")

# Breakdown of missing rows if any
missing_rows = df[df.isna().any(axis=1) | (df == "").any(axis=1)]
if not missing_rows.empty:
    print(f"\n  Details of missing rows ({len(missing_rows)} rows total):")
    for idx, r in missing_rows.iterrows():
        print(f"    - {r['state']} / {r['district']} | {r['class']} | {r['subject']}")

# (4) For each state, class and subject, whether state_score is constant within the state
print("\n" + "=" * 80)
print("(4) CONSTANCY OF STATE_SCORE WITHIN EACH STATE, CLASS, AND SUBJECT")
print("=" * 80)
df_clean_state_score = df.dropna(subset=["state_score"]).copy()
# convert state_score to numeric
df_clean_state_score["state_score_num"] = pd.to_numeric(df_clean_state_score["state_score"], errors="coerce")
state_score_variation = (
    df_clean_state_score.groupby(["state", "class", "subject"])["state_score_num"]
    .agg(n_unique="nunique", unique_vals=lambda x: sorted(x.dropna().unique().tolist()))
)
non_constant = state_score_variation[state_score_variation["n_unique"] > 1]
if non_constant.empty:
    print("  State_score is strictly constant within all states for every class and subject combination.")
else:
    print(f"  Found {len(non_constant)} (state, class, subject) slot(s) where state_score is NOT constant:")
    for (st, cls, subj), row in non_constant.iterrows():
        print(f"    - State: {st}, Class: {cls}, Subject: {subj} -> Distinct values: {row['unique_vals']}")

# (5) Any rows where district_se is above 5 or equal to 0
print("\n" + "=" * 80)
print("(5) ROWS WHERE DISTRICT_SE IS ABOVE 5 OR EQUAL TO 0")
print("=" * 80)
se_numeric = pd.to_numeric(df["district_se"], errors="coerce")
se_anomalies = df[se_numeric.notna() & ((se_numeric > 5) | (se_numeric == 0))].copy()
se_anomalies["district_se_val"] = se_numeric[se_numeric.notna() & ((se_numeric > 5) | (se_numeric == 0))]

if se_anomalies.empty:
    print("  None. All district_se values are > 0 and <= 5.")
else:
    print(f"  Found {len(se_anomalies)} row(s) where district_se > 5 or == 0:")
    for idx, r in se_anomalies.iterrows():
        cond = "> 5" if r["district_se_val"] > 5 else "== 0"
        print(f"    - {r['state']} | {r['district']} | {r['class']} | {r['subject']} | district_se = {r['district_se']} ({cond})")

# (6) Any duplicate combinations of state, district, class, subject
print("\n" + "=" * 80)
print("(6) DUPLICATE COMBINATIONS OF (STATE, DISTRICT, CLASS, SUBJECT)")
print("=" * 80)
dup_keys = ["state", "district", "class", "subject"]
duplicates = df[df.duplicated(subset=dup_keys, keep=False)]
if duplicates.empty:
    print("  None. Every (state, district, class, subject) combination is unique.")
else:
    print(f"  Found {len(duplicates)} duplicate row(s):")
    print(duplicates[dup_keys])
print("=" * 80)
