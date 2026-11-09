import os
from pathlib import Path
import sys
import tempfile

# Ensure matplotlib uses a writable temp cache directory to avoid permission warnings
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_analysis(
    input_path: str = "data/clean/analysis_table.csv",
    output_dir: str = "outputs/analysis",
):
    print("=" * 90)
    print("LEARNING GAP ANALYSIS: NAS 2021 + NFHS-5")
    print("=" * 90)

    # 1. Load data
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"[ERROR] Input file not found: {in_file}")
        sys.exit(1)

    df = pd.read_csv(in_file)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns from '{in_file}'")

    # 2. Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: '{out_dir}/'\n")

    # 3. Check for required columns
    required_cols = [
        "c8_avg",
        "women_10yr",
        "clean_fuel",
        "sanitation",
        "early_marriage",
        "stunting",
        "c8_lang_se",
        "c8_math_se",
        "c8_sci_se",
        "c8_sst_se",
        "state",
        "nas_district",
        "electricity",
        "women_literate",
        "c10_avg",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns in input table: {missing_cols}")
        print("Analysis stopped.")
        sys.exit(1)

    print("All required columns present.\n")

    # 4. Recompute amenities_index as mean of z-scores of clean_fuel and sanitation
    z_clean_fuel = (df["clean_fuel"] - df["clean_fuel"].mean()) / df["clean_fuel"].std()
    z_sanitation = (df["sanitation"] - df["sanitation"].mean()) / df["sanitation"].std()
    df["amenities_index"] = (z_clean_fuel + z_sanitation) / 2.0
    print("Recomputed amenities_index as mean of z-scores of (clean_fuel, sanitation).")

    # Drop rows with missing values in outcome or predictors
    analysis_vars = ["c8_avg", "women_10yr", "amenities_index", "early_marriage", "stunting"]
    initial_rows = len(df)
    df_clean = df.dropna(subset=analysis_vars).copy()
    dropped_rows = initial_rows - len(df_clean)
    print(f"Missing values check: Dropped {dropped_rows} rows with missing values (Retained: {len(df_clean)} rows).\n")

    # 5. Summary statistics, Correlation Matrix, VIF
    print("=" * 90)
    print("1. SUMMARY STATISTICS & PREDICTOR COLLINEARITY")
    print("=" * 90)

    summary_stats = df_clean[analysis_vars].describe().T
    summary_stats["missing"] = df_clean[analysis_vars].isna().sum()
    print("Summary Statistics:")
    print(summary_stats.round(4).to_string())
    summary_stats.to_csv(out_dir / "summary_statistics.csv")
    print(f"Saved: {out_dir / 'summary_statistics.csv'}\n")

    corr_matrix = df_clean[analysis_vars].corr()
    print("Correlation Matrix:")
    print(corr_matrix.round(4).to_string())
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")
    print(f"Saved: {out_dir / 'correlation_matrix.csv'}\n")

    # VIF for predictors
    predictors = ["women_10yr", "amenities_index", "early_marriage", "stunting"]
    X_vif = sm.add_constant(df_clean[predictors])
    vif_records = []
    for i, col in enumerate(predictors):
        val_vif = variance_inflation_factor(X_vif.values, i + 1)
        vif_records.append({"Predictor": col, "VIF": val_vif, "Tolerance (1/VIF)": 1.0 / val_vif})

    vif_df = pd.DataFrame(vif_records)
    print("Variance Inflation Factor (VIF):")
    print(vif_df.round(4).to_string(index=False))
    vif_df.to_csv(out_dir / "vif_table.csv", index=False)
    print(f"Saved: {out_dir / 'vif_table.csv'}\n")

    # 6. Model Estimation
    print("=" * 90)
    print("2. REGRESSION MODELS: MODEL A (OLS HC3) & MODEL B (STATE FE + CLUSTERED SE)")
    print("=" * 90)

    # Model A: OLS with HC3 robust standard errors
    X_A = sm.add_constant(df_clean[predictors])
    y = df_clean["c8_avg"]
    model_A = sm.OLS(y, X_A).fit(cov_type="HC3")

    print("--- MODEL A: OLS with HC3 Robust Standard Errors ---")
    print(f"R-squared: {model_A.rsquared:.4f} | Adj. R-squared: {model_A.rsquared_adj:.4f} | F-stat: {model_A.fvalue:.4f} (p = {model_A.f_pvalue:.4e})")
    print(model_A.summary().tables[1])
    print()

    # Save Model A results table
    model_A_res = pd.DataFrame(
        {
            "Variable": model_A.params.index,
            "Coefficient": model_A.params.values,
            "Std_Error_HC3": model_A.bse.values,
            "t_or_z_stat": model_A.tvalues.values,
            "p_value": model_A.pvalues.values,
            "CI_lower": model_A.conf_int()[0].values,
            "CI_upper": model_A.conf_int()[1].values,
        }
    )
    model_A_res.to_csv(out_dir / "model_A_results.csv", index=False)
    print(f"Saved: {out_dir / 'model_A_results.csv'}\n")

    # Model B: Same plus State Fixed Effects with clustered standard errors by state
    X_B = pd.get_dummies(df_clean[predictors + ["state"]], columns=["state"], drop_first=True, dtype=float)
    X_B = sm.add_constant(X_B)
    model_B = sm.OLS(y, X_B).fit(cov_type="cluster", cov_kwds={"groups": df_clean["state"]})

    print("--- MODEL B: OLS with State Fixed Effects and State-Clustered Standard Errors ---")
    print(f"R-squared: {model_B.rsquared:.4f} | Adj. R-squared: {model_B.rsquared_adj:.4f}")
    print("Core Predictor Coefficients:")
    model_B_pred_df = pd.DataFrame(
        {
            "Coefficient": model_B.params[predictors],
            "Std_Error_Clustered": model_B.bse[predictors],
            "t_or_z_stat": model_B.tvalues[predictors],
            "p_value": model_B.pvalues[predictors],
            "CI_lower": model_B.conf_int().loc[predictors, 0],
            "CI_upper": model_B.conf_int().loc[predictors, 1],
        }
    )
    print(model_B_pred_df.round(4).to_string())
    print()

    # Save full Model B results
    model_B_res = pd.DataFrame(
        {
            "Variable": model_B.params.index,
            "Coefficient": model_B.params.values,
            "Std_Error_Clustered": model_B.bse.values,
            "t_or_z_stat": model_B.tvalues.values,
            "p_value": model_B.pvalues.values,
            "CI_lower": model_B.conf_int()[0].values,
            "CI_upper": model_B.conf_int()[1].values,
        }
    )
    model_B_res.to_csv(out_dir / "model_B_results.csv", index=False)
    print(f"Saved: {out_dir / 'model_B_results.csv'}\n")

    # 7. Residuals & Top / Bottom 15 Districts
    print("=" * 90)
    print("3. RESIDUALS & DISTRICT RANKINGS")
    print("=" * 90)

    df_clean["resid_A"] = model_A.resid
    df_clean["fitted_A"] = model_A.fittedvalues
    df_clean["resid_B"] = model_B.resid
    df_clean["fitted_B"] = model_B.fittedvalues

    # Model A Rankings
    cols_display = ["state", "nas_district", "c8_avg"]

    mod_A_sorted = df_clean.sort_values(by="resid_A", ascending=False).reset_index(drop=True)
    top15_A = mod_A_sorted.head(15)[cols_display + ["fitted_A", "resid_A"]].copy()
    top15_A.insert(0, "rank", range(1, 16))
    bottom15_A = mod_A_sorted.tail(15).iloc[::-1].reset_index(drop=True)[cols_display + ["fitted_A", "resid_A"]].copy()
    bottom15_A.insert(0, "rank", range(1, 16))

    top15_A.to_csv(out_dir / "model_A_top15_resid.csv", index=False)
    bottom15_A.to_csv(out_dir / "model_A_bottom15_resid.csv", index=False)
    print(f"Saved: {out_dir / 'model_A_top15_resid.csv'}")
    print(f"Saved: {out_dir / 'model_A_bottom15_resid.csv'}")

    print("\nModel A - Top 5 Outperforming Districts (Highest Residuals):")
    print(top15_A.head(5).round(3).to_string(index=False))
    print("\nModel A - Bottom 5 Underperforming Districts (Lowest Residuals):")
    print(bottom15_A.head(5).round(3).to_string(index=False))

    # Model B Rankings
    mod_B_sorted = df_clean.sort_values(by="resid_B", ascending=False).reset_index(drop=True)
    top15_B = mod_B_sorted.head(15)[cols_display + ["fitted_B", "resid_B"]].copy()
    top15_B.insert(0, "rank", range(1, 16))
    bottom15_B = mod_B_sorted.tail(15).iloc[::-1].reset_index(drop=True)[cols_display + ["fitted_B", "resid_B"]].copy()
    bottom15_B.insert(0, "rank", range(1, 16))

    top15_B.to_csv(out_dir / "model_B_top15_resid.csv", index=False)
    bottom15_B.to_csv(out_dir / "model_B_bottom15_resid.csv", index=False)
    print(f"Saved: {out_dir / 'model_B_top15_resid.csv'}")
    print(f"Saved: {out_dir / 'model_B_bottom15_resid.csv'}")

    print("\nModel B - Top 5 Outperforming Districts (Highest Residuals):")
    print(top15_B.head(5).round(3).to_string(index=False))
    print("\nModel B - Bottom 5 Underperforming Districts (Lowest Residuals):")
    print(bottom15_B.head(5).round(3).to_string(index=False))
    print()

    # 8. Uncertainty & Significant Gap Flagging
    print("=" * 90)
    print("4. MEASUREMENT UNCERTAINTY & SIGNIFICANT GAP FLAGGING")
    print("=" * 90)

    # Uncertainty formula: sqrt(c8_lang_se^2 + c8_math_se^2 + c8_sci_se^2 + c8_sst_se^2) / 4
    c8_se_approx = np.sqrt(
        df_clean["c8_lang_se"] ** 2
        + df_clean["c8_math_se"] ** 2
        + df_clean["c8_sci_se"] ** 2
        + df_clean["c8_sst_se"] ** 2
    ) / 4.0
    df_clean["c8_avg_se"] = c8_se_approx
    df_clean["uncertainty_threshold"] = 2.0 * c8_se_approx

    # Flag districts where |resid| > 2 * uncertainty
    df_clean["significant_gap_A"] = df_clean["resid_A"].abs() > df_clean["uncertainty_threshold"]
    df_clean["significant_gap_B"] = df_clean["resid_B"].abs() > df_clean["uncertainty_threshold"]

    n_sig_A = df_clean["significant_gap_A"].sum()
    n_sig_B = df_clean["significant_gap_B"].sum()

    print(f"Average Class 8 standard error (c8_avg_se)    : {c8_se_approx.mean():.4f} score points")
    print(f"Average 2-sigma significance threshold (2*SE) : {df_clean['uncertainty_threshold'].mean():.4f} score points")
    print(f"Districts with significant gap (Model A)      : {n_sig_A} / {len(df_clean)} ({n_sig_A / len(df_clean) * 100:.2f}%)")
    print(f"Districts with significant gap (Model B)      : {n_sig_B} / {len(df_clean)} ({n_sig_B / len(df_clean) * 100:.2f}%)")

    # Save significant gap tables
    sig_A_df = df_clean[df_clean["significant_gap_A"]][
        ["state", "nas_district", "c8_avg", "fitted_A", "resid_A", "c8_avg_se", "uncertainty_threshold"]
    ].copy()
    sig_A_df.to_csv(out_dir / "significant_gaps_model_A.csv", index=False)
    print(f"Saved: {out_dir / 'significant_gaps_model_A.csv'}")

    sig_B_df = df_clean[df_clean["significant_gap_B"]][
        ["state", "nas_district", "c8_avg", "fitted_B", "resid_B", "c8_avg_se", "uncertainty_threshold"]
    ].copy()
    sig_B_df.to_csv(out_dir / "significant_gaps_model_B.csv", index=False)
    print(f"Saved: {out_dir / 'significant_gaps_model_B.csv'}\n")

    # 9. Diagnostics (Breusch-Pagan, Residual Plot, Q-Q Plot, Cook's Distance)
    print("=" * 90)
    print("5. REGRESSION DIAGNOSTICS")
    print("=" * 90)

    # Breusch-Pagan test
    bp_lm_stat, bp_lm_pval, bp_f_stat, bp_f_pval = het_breuschpagan(model_A.resid, model_A.model.exog)
    bp_df = pd.DataFrame(
        [
            {"Statistic": "Lagrange Multiplier (LM)", "Value": bp_lm_stat, "p-value": bp_lm_pval},
            {"Statistic": "F-statistic", "Value": bp_f_stat, "p-value": bp_f_pval},
        ]
    )
    print("Breusch-Pagan Test for Heteroskedasticity (Model A):")
    print(bp_df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))
    bp_df.to_csv(out_dir / "breusch_pagan_test.csv", index=False)
    print(f"Saved: {out_dir / 'breusch_pagan_test.csv'}\n")

    # Cook's distance
    influence_A = model_A.get_influence()
    cooks_d = influence_A.cooks_distance[0]
    df_clean["cooks_distance"] = cooks_d

    cooks_top10 = (
        df_clean[["state", "nas_district", "c8_avg", "fitted_A", "resid_A", "cooks_distance"]]
        .sort_values(by="cooks_distance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    cooks_top10.insert(0, "rank", range(1, 11))
    print("Top 10 Districts with Highest Cook's Distance (Model A):")
    print(cooks_top10.round(4).to_string(index=False))
    cooks_top10.to_csv(out_dir / "cooks_distance_top10.csv", index=False)
    print(f"Saved: {out_dir / 'cooks_distance_top10.csv'}\n")

    # Plots
    # Residual vs Fitted
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df_clean["fitted_A"], df_clean["resid_A"], alpha=0.6, color="#1f77b4", edgecolor="none", s=35)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Fitted Values (Predicted c8_avg)", fontsize=11)
    ax.set_ylabel("Residuals (Observed - Predicted)", fontsize=11)
    ax.set_title("Model A: Residuals vs. Fitted Values", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_res_fit_path = out_dir / "residual_vs_fitted.png"
    plt.savefig(plot_res_fit_path)
    plt.close()
    print(f"Saved plot: {plot_res_fit_path}")

    # Q-Q Plot
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sm.qqplot(df_clean["resid_A"], line="45", fit=True, ax=ax, markersize=4, color="#2ca02c", alpha=0.7)
    ax.set_title("Model A: Normal Q-Q Plot of Residuals", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_qq_path = out_dir / "qq_plot.png"
    plt.savefig(plot_qq_path)
    plt.close()
    print(f"Saved plot: {plot_qq_path}\n")

    # 10. Robustness Checks
    print("=" * 90)
    print("6. ROBUSTNESS SPECIFICATIONS")
    print("=" * 90)

    robustness_records = []

    # Baseline Model A
    robustness_records.append(
        {
            "Specification": "1. Baseline Model A (Outcome: c8_avg)",
            "Target_Variable": "women_10yr",
            "Coefficient": model_A.params["women_10yr"],
            "Std_Error": model_A.bse["women_10yr"],
            "t_or_z_stat": model_A.tvalues["women_10yr"],
            "p_value": model_A.pvalues["women_10yr"],
            "R_squared": model_A.rsquared,
            "N_obs": int(model_A.nobs),
        }
    )

    # Robustness 1: Add electricity
    X_rob1 = sm.add_constant(df_clean[predictors + ["electricity"]])
    m_rob1 = sm.OLS(y, X_rob1).fit(cov_type="HC3")
    robustness_records.append(
        {
            "Specification": "2. Model A + Electricity (Outcome: c8_avg)",
            "Target_Variable": "women_10yr",
            "Coefficient": m_rob1.params["women_10yr"],
            "Std_Error": m_rob1.bse["women_10yr"],
            "t_or_z_stat": m_rob1.tvalues["women_10yr"],
            "p_value": m_rob1.pvalues["women_10yr"],
            "R_squared": m_rob1.rsquared,
            "N_obs": int(m_rob1.nobs),
        }
    )

    # Robustness 2: Replace women_10yr with women_literate
    pred_rob2 = ["women_literate", "amenities_index", "early_marriage", "stunting"]
    X_rob2 = sm.add_constant(df_clean[pred_rob2])
    m_rob2 = sm.OLS(y, X_rob2).fit(cov_type="HC3")
    robustness_records.append(
        {
            "Specification": "3. Replace women_10yr with women_literate (Outcome: c8_avg)",
            "Target_Variable": "women_literate",
            "Coefficient": m_rob2.params["women_literate"],
            "Std_Error": m_rob2.bse["women_literate"],
            "t_or_z_stat": m_rob2.tvalues["women_literate"],
            "p_value": m_rob2.pvalues["women_literate"],
            "R_squared": m_rob2.rsquared,
            "N_obs": int(m_rob2.nobs),
        }
    )

    # Robustness 3: Replace c8_avg with c10_avg outcome
    y_c10 = df_clean["c10_avg"]
    m_rob3 = sm.OLS(y_c10, X_A).fit(cov_type="HC3")
    robustness_records.append(
        {
            "Specification": "4. Replace c8_avg outcome with c10_avg",
            "Target_Variable": "women_10yr",
            "Coefficient": m_rob3.params["women_10yr"],
            "Std_Error": m_rob3.bse["women_10yr"],
            "t_or_z_stat": m_rob3.tvalues["women_10yr"],
            "p_value": m_rob3.pvalues["women_10yr"],
            "R_squared": m_rob3.rsquared,
            "N_obs": int(m_rob3.nobs),
        }
    )

    rob_df = pd.DataFrame(robustness_records)
    print("Robustness Checks Summary:")
    print(rob_df.round(4).to_string(index=False))
    rob_df.to_csv(out_dir / "robustness_checks.csv", index=False)
    print(f"Saved: {out_dir / 'robustness_checks.csv'}\n")

    # 11. Variance Decomposition (Between vs. Within State)
    print("=" * 90)
    print("7. VARIANCE DECOMPOSITION OF c8_avg (BETWEEN VS. WITHIN STATES)")
    print("=" * 90)

    grand_mean = y.mean()
    state_means = df_clean.groupby("state")["c8_avg"].transform("mean")

    ss_total = np.sum((y - grand_mean) ** 2)
    ss_between = np.sum((state_means - grand_mean) ** 2)
    ss_within = np.sum((y - state_means) ** 2)

    share_between = ss_between / ss_total
    share_within = ss_within / ss_total

    var_decomp_df = pd.DataFrame(
        [
            {
                "Component": "Between States",
                "Sum_of_Squares": ss_between,
                "Variance_Share": share_between,
                "Percentage": f"{share_between * 100:.2f}%",
            },
            {
                "Component": "Within States",
                "Sum_of_Squares": ss_within,
                "Variance_Share": share_within,
                "Percentage": f"{share_within * 100:.2f}%",
            },
            {
                "Component": "Total",
                "Sum_of_Squares": ss_total,
                "Variance_Share": 1.0,
                "Percentage": "100.00%",
            },
        ]
    )

    print("Variance Decomposition:")
    print(var_decomp_df.to_string(index=False))
    var_decomp_df.to_csv(out_dir / "variance_decomposition.csv", index=False)
    print(f"Saved: {out_dir / 'variance_decomposition.csv'}\n")

    print("=" * 90)
    print("ANALYSIS COMPLETE: All tables and plots generated successfully.")
    print("=" * 90)


if __name__ == "__main__":
    run_analysis()
