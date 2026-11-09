import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    nas_path = Path("data/clean/nas_district_wide_v2.csv")
    at_path = Path("data/clean/analysis_table.csv")
    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nas = pd.read_csv(nas_path)
    df_at = pd.read_csv(at_path)

    # Recompute amenities_index as mean of z-scores of clean_fuel and sanitation
    z_clean_fuel = (df_at["clean_fuel"] - df_at["clean_fuel"].mean()) / df_at["clean_fuel"].std()
    z_sanitation = (df_at["sanitation"] - df_at["sanitation"].mean()) / df_at["sanitation"].std()
    df_at["amenities_index"] = (z_clean_fuel + z_sanitation) / 2.0

    # c8_avg calculation for df_nas if not already present
    if "c8_avg" not in df_nas.columns:
        df_nas["c8_avg"] = (df_nas["c8_lang"] + df_nas["c8_math"] + df_nas["c8_sci"] + df_nas["c8_sst"]) / 4.0

    print("=" * 110)
    print("PART 1: 15 NAS SCORES & SEs FOR SPECIFIED DISTRICTS + STATE c8_avg MEAN")
    print("=" * 110)

    target_districts = [
        ("Rajasthan", "Dhaulpur"),
        ("Madhya Pradesh", "Bhind"),
        ("Madhya Pradesh", "Damoh"),
        ("Madhya Pradesh", "Sidhi"),
        ("Punjab", "Barnala"),
    ]

    scores_list = [
        "c3_lang", "c3_math", "c3_evs",
        "c5_lang", "c5_math", "c5_evs",
        "c8_lang", "c8_math", "c8_sci", "c8_sst",
        "c10_lang", "c10_math", "c10_sci", "c10_sst", "c10_eng"
    ]

    # Precompute state means
    state_c8_means_nas = df_nas.groupby("state")["c8_avg"].mean().to_dict()
    state_c8_means_at = df_at.groupby("state")["c8_avg"].mean().to_dict()

    p1_rows = []
    for st, dt in target_districts:
        row_nas = df_nas[(df_nas["state"] == st) & (df_nas["district"] == dt)].iloc[0]
        row_at = df_at[(df_at["state"] == st) & (df_at["nas_district"] == dt)].iloc[0]
        
        d_dict = {
            "State": st,
            "District": dt,
            "District_c8_avg": row_at["c8_avg"],
            "State_c8_avg_mean (NAS)": state_c8_means_nas[st],
            "State_c8_avg_mean (Analysis Table)": state_c8_means_at[st],
        }
        for sc in scores_list:
            d_dict[f"{sc}"] = row_nas[sc]
            d_dict[f"{sc}_se"] = row_nas[f"{sc}_se"]
        p1_rows.append(d_dict)

    df_p1 = pd.DataFrame(p1_rows)
    # Print formatted per district
    for _, r in df_p1.iterrows():
        print(f"\n--- {r['State']} - {r['District']} ---")
        print(f"District c8_avg: {r['District_c8_avg']:.2f} | State c8_avg Mean (NAS, N_districts={len(df_nas[df_nas.state == r['State']])}): {r['State_c8_avg_mean (NAS)']:.4f} | State c8_avg Mean (Analysis Table, N={len(df_at[df_at.state == r['State']])}): {r['State_c8_avg_mean (Analysis Table)']:.4f}")
        subj_data = []
        for sc in scores_list:
            subj_data.append({
                "Subject": sc,
                "Score": f"{r[sc]:.2f}",
                "SE": f"{r[sc + '_se']:.2f}"
            })
        print(pd.DataFrame(subj_data).to_string(index=False))

    print("\n" + "=" * 110)
    print("PART 2: RE-RUN MODEL A & MODEL B WITHOUT DHAULPUR VS. FULL SAMPLE")
    print("=" * 110)

    predictors = ["women_10yr", "amenities_index", "early_marriage", "stunting"]
    
    # 1. Full Sample Models
    y_full = df_at["c8_avg"]
    X_A_full = sm.add_constant(df_at[predictors])
    mA_full = sm.OLS(y_full, X_A_full).fit(cov_type="HC3")

    X_B_full = pd.get_dummies(df_at[predictors + ["state"]], columns=["state"], drop_first=True, dtype=float)
    X_B_full = sm.add_constant(X_B_full)
    mB_full = sm.OLS(y_full, X_B_full).fit(cov_type="cluster", cov_kwds={"groups": df_at["state"]})

    # 2. Excl Dhaulpur Models
    df_no_dh = df_at[~((df_at["state"] == "Rajasthan") & (df_at["nas_district"] == "Dhaulpur"))].copy()
    y_no_dh = df_no_dh["c8_avg"]
    X_A_no_dh = sm.add_constant(df_no_dh[predictors])
    mA_no_dh = sm.OLS(y_no_dh, X_A_no_dh).fit(cov_type="HC3")

    X_B_no_dh = pd.get_dummies(df_no_dh[predictors + ["state"]], columns=["state"], drop_first=True, dtype=float)
    X_B_no_dh = sm.add_constant(X_B_no_dh)
    mB_no_dh = sm.OLS(y_no_dh, X_B_no_dh).fit(cov_type="cluster", cov_kwds={"groups": df_no_dh["state"]})

    # Model A Side-by-Side Comparison
    comp_A = []
    for var in ["const"] + predictors:
        comp_A.append({
            "Variable": var,
            "Full_Coef": mA_full.params[var],
            "Full_SE_HC3": mA_full.bse[var],
            "Full_p": mA_full.pvalues[var],
            "NoDh_Coef": mA_no_dh.params[var],
            "NoDh_SE_HC3": mA_no_dh.bse[var],
            "NoDh_p": mA_no_dh.pvalues[var],
        })
    df_comp_A = pd.DataFrame(comp_A)
    print("\n--- MODEL A: Full Sample (N=683, R²={:.4f}) vs. Without Dhaulpur (N=682, R²={:.4f}) ---".format(mA_full.rsquared, mA_no_dh.rsquared))
    print(df_comp_A.to_string(index=False, formatters={
        "Full_Coef": "{:.4f}".format, "Full_SE_HC3": "{:.4f}".format, "Full_p": "{:.4e}".format,
        "NoDh_Coef": "{:.4f}".format, "NoDh_SE_HC3": "{:.4f}".format, "NoDh_p": "{:.4e}".format,
    }))

    # Model B Side-by-Side Comparison (Core Predictors)
    comp_B = []
    for var in predictors:
        comp_B.append({
            "Variable": var,
            "Full_Coef": mB_full.params[var],
            "Full_SE_Clustered": mB_full.bse[var],
            "Full_p": mB_full.pvalues[var],
            "NoDh_Coef": mB_no_dh.params[var],
            "NoDh_SE_Clustered": mB_no_dh.bse[var],
            "NoDh_p": mB_no_dh.pvalues[var],
        })
    df_comp_B = pd.DataFrame(comp_B)
    print("\n--- MODEL B: Full Sample (N=683, R²={:.4f}) vs. Without Dhaulpur (N=682, R²={:.4f}) ---".format(mB_full.rsquared, mB_no_dh.rsquared))
    print(df_comp_B.to_string(index=False, formatters={
        "Full_Coef": "{:.4f}".format, "Full_SE_Clustered": "{:.4f}".format, "Full_p": "{:.4e}".format,
        "NoDh_Coef": "{:.4f}".format, "NoDh_SE_Clustered": "{:.4f}".format, "NoDh_p": "{:.4e}".format,
    }))

    print("\n" + "=" * 110)
    print("PART 3: RE-RUN MODEL A & MODEL B WITHOUT EARLY_MARRIAGE")
    print("=" * 110)

    pred_no_em = ["women_10yr", "amenities_index", "stunting"]

    # Model A without early_marriage
    X_A_no_em = sm.add_constant(df_at[pred_no_em])
    mA_no_em = sm.OLS(y_full, X_A_no_em).fit(cov_type="HC3")

    # Model B without early_marriage
    X_B_no_em = pd.get_dummies(df_at[pred_no_em + ["state"]], columns=["state"], drop_first=True, dtype=float)
    X_B_no_em = sm.add_constant(X_B_no_em)
    mB_no_em = sm.OLS(y_full, X_B_no_em).fit(cov_type="cluster", cov_kwds={"groups": df_at["state"]})

    res_no_em = []
    for var in pred_no_em:
        res_no_em.append({
            "Variable": var,
            "Model_A_Coef": mA_no_em.params[var],
            "Model_A_SE_HC3": mA_no_em.bse[var],
            "Model_A_t": mA_no_em.tvalues[var],
            "Model_A_p": mA_no_em.pvalues[var],
            "Model_B_Coef": mB_no_em.params[var],
            "Model_B_SE_Clust": mB_no_em.bse[var],
            "Model_B_t": mB_no_em.tvalues[var],
            "Model_B_p": mB_no_em.pvalues[var],
        })
    df_res_no_em = pd.DataFrame(res_no_em)
    print(f"Model A (No early_marriage) R²: {mA_no_em.rsquared:.4f} | Model B (No early_marriage) R²: {mB_no_em.rsquared:.4f}\n")
    print(df_res_no_em.to_string(index=False, formatters={
        "Model_A_Coef": "{:.4f}".format, "Model_A_SE_HC3": "{:.4f}".format, "Model_A_t": "{:.4f}".format, "Model_A_p": "{:.4e}".format,
        "Model_B_Coef": "{:.4f}".format, "Model_B_SE_Clust": "{:.4f}".format, "Model_B_t": "{:.4f}".format, "Model_B_p": "{:.4e}".format,
    }))

    print("\n" + "=" * 110)
    print("PART 4: RESIDUALS & DECILE FLAGS -> outputs/analysis/district_results.csv")
    print("=" * 110)

    # Compute resid_A and resid_B from full models
    df_at["resid_A"] = mA_full.resid
    df_at["resid_B"] = mB_full.resid

    # Top decile (top 10% highest resid_B) and bottom decile (bottom 10% lowest resid_B)
    q10 = df_at["resid_B"].quantile(0.10)
    q90 = df_at["resid_B"].quantile(0.90)

    df_at["top_decile"] = df_at["resid_B"] >= q90
    df_at["bottom_decile"] = df_at["resid_B"] <= q10

    print(f"Threshold for bottom decile (resid_B <= {q10:.4f}): {df_at['bottom_decile'].sum()} districts")
    print(f"Threshold for top decile (resid_B >= {q90:.4f}): {df_at['top_decile'].sum()} districts")

    # Save to outputs/analysis/district_results.csv
    out_csv = out_dir / "district_results.csv"
    df_at.to_csv(out_csv, index=False)
    print(f"Saved full district results to: '{out_csv}' ({len(df_at)} rows, {len(df_at.columns)} columns)")

    print("\n" + "=" * 110)
    print("PART 5: 10 STATES WITH HIGHEST & LOWEST MEAN resid_A")
    print("=" * 110)

    state_resid = df_at.groupby("state")["resid_A"].agg(
        n_districts="count",
        mean_resid_A="mean",
        std_resid_A="std",
        min_resid_A="min",
        max_resid_A="max"
    ).reset_index()

    top10_states = state_resid.sort_values(by="mean_resid_A", ascending=False).head(10).reset_index(drop=True)
    bottom10_states = state_resid.sort_values(by="mean_resid_A", ascending=True).head(10).reset_index(drop=True)

    print("\n--- 10 States with HIGHEST Mean resid_A ---")
    print(top10_states.to_string(index=False, formatters={
        "mean_resid_A": "{:.4f}".format, "std_resid_A": "{:.4f}".format,
        "min_resid_A": "{:.4f}".format, "max_resid_A": "{:.4f}".format
    }))

    print("\n--- 10 States with LOWEST Mean resid_A ---")
    print(bottom10_states.to_string(index=False, formatters={
        "mean_resid_A": "{:.4f}".format, "std_resid_A": "{:.4f}".format,
        "min_resid_A": "{:.4f}".format, "max_resid_A": "{:.4f}".format
    }))

if __name__ == "__main__":
    main()
