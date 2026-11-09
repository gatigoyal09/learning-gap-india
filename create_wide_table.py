import pandas as pd
from pathlib import Path

# File paths
input_path = Path("data/clean/nas_district_scores.csv")
output_path = Path("data/clean/nas_district_wide.csv")

# Load long data
df = pd.read_csv(input_path)

# Class and subject mappings
class_map = {
    "Class 3": "c3",
    "Class 5": "c5",
    "Class 8": "c8",
    "Class 10": "c10"
}

subj_map = {
    "Language": "lang",
    "Maths": "math",
    "EVS": "evs",
    "Science": "sci",
    "Social Science": "sst",
    "English": "eng"
}

# Create column base name (e.g. c8_math, c10_eng)
df["col_base"] = df["class"].map(class_map) + "_" + df["subject"].map(subj_map)

# Pivot scores and SEs
pivot_scores = df.pivot(index=["state", "district"], columns="col_base", values="district_score")
pivot_se = df.pivot(index=["state", "district"], columns="col_base", values="district_se")
pivot_se.columns = [c + "_se" for c in pivot_se.columns]

# Combine into wide format
wide = pd.concat([pivot_scores, pivot_se], axis=1).reset_index()

# Ordered score and SE columns
score_order = [
    "c3_lang", "c3_math", "c3_evs",
    "c5_lang", "c5_math", "c5_evs",
    "c8_lang", "c8_math", "c8_sci", "c8_sst",
    "c10_lang", "c10_math", "c10_sci", "c10_sst", "c10_eng"
]
se_order = [c + "_se" for c in score_order]
c8_cols = ["c8_lang", "c8_math", "c8_sci", "c8_sst"]

# Compute avg_score (mean of available scores) and c8_avg (mean of Class 8 subjects)
wide["avg_score"] = wide[score_order].mean(axis=1)
wide["c8_avg"] = wide[c8_cols].mean(axis=1)

# Organize final column layout
final_cols = ["state", "district"] + score_order + se_order + ["avg_score", "c8_avg"]
wide = wide[final_cols]

# Sort alphabetically by state and district
wide = wide.sort_values(by=["state", "district"]).reset_index(drop=True)

# Save to CSV
output_path.parent.mkdir(parents=True, exist_ok=True)
wide.to_csv(output_path, index=False)

print(f"Saved wide table to: {output_path}")
print(f"Shape: {wide.shape} (rows, columns)\n")

# Display first 10 rows
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print("First 10 rows:")
print(wide.head(10))
