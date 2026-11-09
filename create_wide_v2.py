import pandas as pd
from pathlib import Path

# Paths
wide_input_path = Path("data/clean/nas_district_wide.csv")
long_input_path = Path("data/clean/nas_district_scores.csv")
output_path = Path("data/clean/nas_district_wide_v2.csv")

# 1. Load wide and long tables
wide = pd.read_csv(wide_input_path)
long = pd.read_csv(long_input_path)

# 2. State cleaning function:
#    - replace underscores with spaces
#    - fix Chattisgarh to Chhattisgarh
#    - ensure Jammu & Kashmir has proper spacing
def clean_state(name: str) -> str:
    cleaned = name.replace("_", " ")
    if cleaned == "Chattisgarh":
        cleaned = "Chhattisgarh"
    return cleaned

wide["state"] = wide["state"].apply(clean_state)
long["state"] = long["state"].apply(clean_state)

# 3. Add class averages to wide table
c3_cols = ["c3_lang", "c3_math", "c3_evs"]
c5_cols = ["c5_lang", "c5_math", "c5_evs"]
c8_cols = ["c8_lang", "c8_math", "c8_sci", "c8_sst"]
c10_cols = ["c10_lang", "c10_math", "c10_sci", "c10_sst", "c10_eng"]

wide["c3_avg"] = wide[c3_cols].mean(axis=1)
wide["c5_avg"] = wide[c5_cols].mean(axis=1)
# c8_avg was already computed, recompute to be certain
wide["c8_avg"] = wide[c8_cols].mean(axis=1)
wide["c10_avg"] = wide[c10_cols].mean(axis=1)

# 4. Add state_c8_avg column holding the state-level Class 8 average (from state_score in long file)
c8_long = long[long["class"] == "Class 8"].dropna(subset=["state_score"])
state_c8_map = (
    c8_long.drop_duplicates(subset=["state", "subject"])
    .groupby("state")["state_score"]
    .mean()
    .rename("state_c8_avg")
)

wide = wide.merge(state_c8_map, on="state", how="left")

# 5. Define column order
score_order = [
    "c3_lang", "c3_math", "c3_evs",
    "c5_lang", "c5_math", "c5_evs",
    "c8_lang", "c8_math", "c8_sci", "c8_sst",
    "c10_lang", "c10_math", "c10_sci", "c10_sst", "c10_eng"
]
se_order = [c + "_se" for c in score_order]
avg_order = ["c3_avg", "c5_avg", "c8_avg", "c10_avg", "avg_score", "state_c8_avg"]

final_cols = ["state", "district"] + score_order + se_order + avg_order
wide = wide[final_cols]

# Sort alphabetically by state, then district
wide = wide.sort_values(by=["state", "district"]).reset_index(drop=True)

# 6. Save to data/clean/nas_district_wide_v2.csv
output_path.parent.mkdir(parents=True, exist_ok=True)
wide.to_csv(output_path, index=False)

print(f"Successfully created: {output_path}")
print(f"Shape: {wide.shape} (rows, columns)\n")

# Print unique state names
unique_states = sorted(wide["state"].unique())
print(f"List of Unique State Names ({len(unique_states)} total):")
print("-" * 40)
for idx, st in enumerate(unique_states, 1):
    print(f"{idx:>2}. {st}")
print("-" * 40)
