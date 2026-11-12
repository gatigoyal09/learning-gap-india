from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="The Learning Gap in India: District Learning Outcomes and Household Conditions",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #111827;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .plain-reading {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #1E40AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data_dir = Path(__file__).parent / "data"

    df_district = pd.read_csv(data_dir / "district_results.csv")
    if "fitted_B" not in df_district.columns:
        df_district["fitted_B"] = df_district["c8_avg"] - df_district["resid_B"]
    if "fitted_A" not in df_district.columns:
        df_district["fitted_A"] = df_district["c8_avg"] - df_district["resid_A"]

    df_model_a = pd.read_csv(data_dir / "model_A_results.csv")
    df_model_b = pd.read_csv(data_dir / "model_B_results.csv")
    df_robustness = pd.read_csv(data_dir / "robustness_checks.csv")
    df_variance = pd.read_csv(data_dir / "variance_decomposition.csv")

    return df_district, df_model_a, df_model_b, df_robustness, df_variance


df_district, df_model_a, df_model_b, df_robustness, df_variance = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["Overview", "States", "District Explorer", "Models", "Method & Limitations"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data sources: NAS 2021 & NFHS-5 (2019-21)")


# -------------------------------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="main-title">The Learning Gap in India: District Learning Outcomes and Household Conditions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">An empirical investigation reconciling National Achievement Survey (NAS 2021) '
        'learning outcomes with National Family Health Survey (NFHS-5) socio-economic indicators across Indian districts.</div>',
        unsafe_allow_html=True,
    )

    n_districts = len(df_district)
    n_states = df_district["state"].nunique()

    # Sample info badges
    st.info(f"📍 **Dataset Coverage**: **{n_districts}** harmonised districts across **{n_states}** States and Union Territories.")

    # 3 Headline Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">VARIANCE EXPLAINED BY GEOGRAPHY</div>
                <div class="metric-value">55%</div>
                <div class="metric-label">of variance in Class 8 scores lies between states</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">MODEL A (NO FIXED EFFECTS)</div>
                <div class="metric-value">R² = 0.08</div>
                <div class="metric-label">District socio-economic predictors alone explain 8%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">MODEL B (WITH STATE FIXED EFFECTS)</div>
                <div class="metric-value">R² = 0.62</div>
                <div class="metric-label">State-level differences lift explanatory power to 62%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Variance Decomposition Bar Chart
    st.subheader("Variance Decomposition: Class 8 Average Score")
    st.markdown(
        "Decomposition of the total sum of squares of district Class 8 average test scores (`c8_avg`) into "
        "variation **between states** versus variation **within states**."
    )

    # Clean display for variance decomposition
    df_var_plot = df_variance[df_variance["Component"] != "Total"].copy()
    df_var_plot["Share_Pct"] = df_var_plot["Variance_Share"] * 100

    fig_var = px.bar(
        df_var_plot,
        x="Component",
        y="Share_Pct",
        text=df_var_plot["Share_Pct"].apply(lambda v: f"{v:.1f}%"),
        color="Component",
        color_discrete_map={"Between States": "#2563EB", "Within States": "#93C5FD"},
        labels={"Share_Pct": "Share of Total Variance (%)", "Component": "Variance Component"},
        title="Share of Class 8 Score Variance (Between vs. Within States)",
    )
    fig_var.update_layout(
        yaxis_range=[0, 70],
        showlegend=False,
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig_var.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Share: %{y:.2f}%<extra></extra>",
    )
    st.plotly_chart(fig_var, use_container_width=True)

    with st.expander("View Variance Decomposition Data Table"):
        st.dataframe(df_variance.style.format({"Sum_of_Squares": "{:,.2f}", "Variance_Share": "{:.4f}"}), use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 2: STATES
# -------------------------------------------------------------------------------------------------
elif page == "States":
    st.title("State-Level Performance & Residuals")
    st.markdown(
        "Compare average Class 8 learning scores across states, as well as mean Model A residuals "
        "(actual performance above or below what district socio-economic predictors predict)."
    )

    # Compute state level aggregated statistics
    state_agg = df_district.groupby("state").agg(
        n_districts=("nas_district", "count"),
        mean_c8_avg=("c8_avg", "mean"),
        mean_resid_A=("resid_A", "mean"),
        mean_women_10yr=("women_10yr", "mean"),
        mean_stunting=("stunting", "mean"),
    ).reset_index()

    # Chart 1: Mean c8_avg by state
    st.subheader("1. Mean Class 8 Score (`c8_avg`) by State")
    c1_sort = st.selectbox(
        "Sort Class 8 score chart by:",
        ["Mean Score (Highest to Lowest)", "Mean Score (Lowest to Highest)", "State Name (A to Z)"],
        key="sort_c8",
    )

    if c1_sort == "Mean Score (Highest to Lowest)":
        df_c8_sorted = state_agg.sort_values(by="mean_c8_avg", ascending=False)
    elif c1_sort == "Mean Score (Lowest to Highest)":
        df_c8_sorted = state_agg.sort_values(by="mean_c8_avg", ascending=True)
    else:
        df_c8_sorted = state_agg.sort_values(by="state", ascending=True)

    fig_c8 = px.bar(
        df_c8_sorted,
        x="state",
        y="mean_c8_avg",
        custom_data=["n_districts", "mean_women_10yr", "mean_stunting"],
        labels={"state": "State / UT", "mean_c8_avg": "Mean Class 8 Score"},
        color="mean_c8_avg",
        color_continuous_scale="Blues",
        title="Average Class 8 Score by State (NAS 2021)",
    )
    fig_c8.update_traces(
        hovertemplate="<b>%{x}</b><br>"
        + "Mean Class 8 Score: %{y:.2f}<br>"
        + "Districts in Sample: %{customdata[0]}<br>"
        + "Mean Women 10+ yr Schooling: %{customdata[1]:.1f}%<br>"
        + "Mean Child Stunting: %{customdata[2]:.1f}%"
        + "<extra></extra>"
    )
    fig_c8.update_layout(
        template="plotly_white",
        height=480,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=100),
    )
    st.plotly_chart(fig_c8, use_container_width=True)

    st.markdown("---")

    # Chart 2: Mean resid_A by state
    st.subheader("2. Mean Model A Residual (`resid_A`) by State")
    st.caption("Positive values indicate state performance is above predicted; negative values indicate performance is below predicted by socio-economic predictors.")

    c2_sort = st.selectbox(
        "Sort Model A residual chart by:",
        ["Mean Residual (Highest to Lowest)", "Mean Residual (Lowest to Highest)", "State Name (A to Z)"],
        key="sort_resid",
    )

    if c2_sort == "Mean Residual (Highest to Lowest)":
        df_resid_sorted = state_agg.sort_values(by="mean_resid_A", ascending=False)
    elif c2_sort == "Mean Residual (Lowest to Highest)":
        df_resid_sorted = state_agg.sort_values(by="mean_resid_A", ascending=True)
    else:
        df_resid_sorted = state_agg.sort_values(by="state", ascending=True)

    # Color code positive (above predicted) vs negative (below predicted)
    df_resid_sorted["residual_status"] = df_resid_sorted["mean_resid_A"].apply(
        lambda r: "Above Predicted" if r >= 0 else "Below Predicted"
    )

    fig_resid = px.bar(
        df_resid_sorted,
        x="state",
        y="mean_resid_A",
        custom_data=["n_districts"],
        color="residual_status",
        color_discrete_map={"Above Predicted": "#10B981", "Below Predicted": "#EF4444"},
        labels={"state": "State / UT", "mean_resid_A": "Mean Model A Residual"},
        title="Mean Model A Residual by State (Above / Below Predicted)",
    )
    fig_resid.update_traces(
        hovertemplate="<b>%{x}</b><br>"
        + "Mean Residual: %{y:.2f} score points<br>"
        + "Districts in Sample: %{customdata[0]}<br>"
        + "<extra></extra>"
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="#374151", line_width=1)
    fig_resid.update_layout(
        template="plotly_white",
        height=480,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=100),
        legend_title_text="Status",
    )
    st.plotly_chart(fig_resid, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 3: DISTRICT EXPLORER
# -------------------------------------------------------------------------------------------------
elif page == "District Explorer":
    st.title("District Explorer")
    st.markdown(
        "Explore district-level outcomes, compare actual vs. predicted Class 8 performance from Model B, "
        "and inspect districts flagged in the top and bottom deciles."
    )

    # Create category for decile flags
    def get_decile_status(row):
        if row["top_decile"]:
            return "Top Decile (Above Predicted)"
        elif row["bottom_decile"]:
            return "Bottom Decile (Below Predicted)"
        return "Middle 80%"

    df_district_exp = df_district.copy()
    df_district_exp["decile_status"] = df_district_exp.apply(get_decile_status, axis=1)

    # Filter controls
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        selected_states = st.multiselect(
            "Filter by State / UT (leave blank for all):",
            options=sorted(df_district_exp["state"].unique()),
            default=[],
        )
    with col_f2:
        search_district = st.text_input("Search District Name:", placeholder="e.g. Dhaulpur, Bhind, Barnala...")

    # Apply filters
    filtered_df = df_district_exp
    if selected_states:
        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
    if search_district.strip():
        filtered_df = filtered_df[
            filtered_df["nas_district"].str.contains(search_district.strip(), case=False, na=False)
        ]

    st.caption(f"Displaying **{len(filtered_df)}** of {len(df_district_exp)} districts")

    # Scatter plot: Actual vs Predicted c8_avg
    st.subheader("Actual vs. Predicted Class 8 Score (Model B)")
    fig_scatter = px.scatter(
        filtered_df,
        x="fitted_B",
        y="c8_avg",
        color="decile_status",
        color_discrete_map={
            "Top Decile (Above Predicted)": "#10B981",
            "Bottom Decile (Below Predicted)": "#EF4444",
            "Middle 80%": "#6B7280",
        },
        hover_name="nas_district",
        hover_data={
            "state": True,
            "c8_avg": ":.2f",
            "fitted_B": ":.2f",
            "resid_B": ":.2f",
            "decile_status": False,
        },
        labels={
            "fitted_B": "Predicted Class 8 Score (Model B)",
            "c8_avg": "Actual Class 8 Score",
            "decile_status": "Decile Group",
        },
        title="Actual vs. Predicted Class 8 Score by District",
    )
    # Add 45 degree line
    min_val = min(df_district_exp["fitted_B"].min(), df_district_exp["c8_avg"].min()) - 2
    max_val = max(df_district_exp["fitted_B"].max(), df_district_exp["c8_avg"].max()) + 2
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            name="Actual = Predicted (45° line)",
        )
    )
    fig_scatter.update_layout(
        template="plotly_white",
        height=520,
        legend_title_text="Performance Status",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # District Table sorted by resid_B
    st.subheader("District Data Table (Sorted by Model B Residual)")
    display_cols = [
        "state",
        "nas_district",
        "c8_avg",
        "fitted_B",
        "resid_B",
        "top_decile",
        "bottom_decile",
    ]

    table_df = filtered_df[display_cols].sort_values(by="resid_B", ascending=False).reset_index(drop=True)
    table_df = table_df.rename(
        columns={
            "nas_district": "district",
            "fitted_B": "predicted_value_model_B",
        }
    )

    st.dataframe(
        table_df.style.format(
            {
                "c8_avg": "{:.2f}",
                "predicted_value_model_B": "{:.2f}",
                "resid_B": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=400,
    )


# -------------------------------------------------------------------------------------------------
# PAGE 4: MODELS
# -------------------------------------------------------------------------------------------------
elif page == "Models":
    st.title("Regression Models & Robustness Specifications")
    st.markdown(
        "Compare baseline Model A (OLS with HC3 robust standard errors) with Model B "
        "(State Fixed Effects with state-clustered standard errors), along with robustness checks."
    )

    # Side-by-side Model A and Model B
    st.subheader("Model A vs. Model B: Baseline Comparison")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Model A: OLS with HC3 Robust SE")
        st.caption("Outcome: `c8_avg` | Sample: 683 districts | R²: 0.0803")
        mod_a_show = df_model_a.copy()
        st.dataframe(
            mod_a_show.style.format(
                {
                    "Coefficient": "{:.4f}",
                    "Std_Error_HC3": "{:.4f}",
                    "t_or_z_stat": "{:.3f}",
                    "p_value": "{:.4e}",
                    "CI_lower": "{:.4f}",
                    "CI_upper": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    with col_b:
        st.markdown("#### Model B: State Fixed Effects & Clustered SE")
        st.caption("Outcome: `c8_avg` | Sample: 683 districts | R²: 0.6172 | Clustered by State")
        # Filter to core predictors for clean comparison
        core_vars = ["women_10yr", "amenities_index", "early_marriage", "stunting"]
        mod_b_core = df_model_b[df_model_b["Variable"].isin(core_vars)].copy()
        st.dataframe(
            mod_b_core.style.format(
                {
                    "Coefficient": "{:.4f}",
                    "Std_Error_Clustered": "{:.4f}",
                    "t_or_z_stat": "{:.3f}",
                    "p_value": "{:.4e}",
                    "CI_lower": "{:.4f}",
                    "CI_upper": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown(
        '<div class="plain-reading">'
        "<b>Plain-English Reading:</b> While district socio-economic factors alone explain only ~8% of the cross-district learning variation in Model A, "
        "introducing state fixed effects increases explanatory power to ~62% in Model B, demonstrating that institutional and state-level policy "
        "differences are the predominant driver of tested student outcomes."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View Full Model B State Fixed Effect Coefficients"):
        st.dataframe(
            df_model_b.style.format(
                {
                    "Coefficient": "{:.4f}",
                    "Std_Error_Clustered": "{:.4f}",
                    "t_or_z_stat": "{:.3f}",
                    "p_value": "{:.4e}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # Robustness Table
    st.subheader("Robustness Checks")
    st.caption("Evaluating stability of female education effects across model alterations and grade levels.")

    rob_display = df_robustness.copy()
    st.dataframe(
        rob_display.style.format(
            {
                "Coefficient": "{:.4f}",
                "Std_Error": "{:.4f}",
                "t_or_z_stat": "{:.3f}",
                "p_value": "{:.4e}",
                "R_squared": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        '<div class="plain-reading">'
        "<b>Plain-English Reading:</b> Across all robustness specifications—controlling for household electricity, substituting female literacy for 10+ years schooling, "
        "and evaluating Class 10 rather than Class 8 scores—the positive relationship between women's education and learning outcomes remains statistically significant and stable."
        "</div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------------------------------
# PAGE 5: METHOD & LIMITATIONS
# -------------------------------------------------------------------------------------------------
elif page == "Method & Limitations":
    st.title("Methodology & Analytical Limitations")

    # Planned vs delivered
    st.subheader("Planned vs delivered")
    df_planned = pd.DataFrame(
        [
            {
                "Item": "Educational resources",
                "Originally planned": "School resource indicators from UDISE+",
                "What was done": "District-level UDISE+ data was not available in usable form, so NFHS-5 household indicators were used instead",
            },
            {
                "Item": "Research question",
                "Originally planned": "How efficiently districts convert resources into learning",
                "What was done": "How district learning compares with what household conditions predict (above or below predicted)",
            },
            {
                "Item": "Multilevel modelling",
                "Originally planned": "Districts nested within states",
                "What was done": "State fixed effects with state-clustered standard errors, plus a between/within-state variance decomposition",
            },
            {
                "Item": "Inequality decomposition",
                "Originally planned": "Gini/Theil",
                "What was done": "Between-state vs within-state variance decomposition (55% / 45%)",
            },
            {
                "Item": "DEA / SFA",
                "Originally planned": "Frontier efficiency models",
                "What was done": "Not applicable: they need school inputs, which were not available",
            },
            {
                "Item": "Spatial analysis",
                "Originally planned": "Moran's I and spatial regression",
                "What was done": "Not done (future work); state-level clustering of residuals is shown, but no spatial test was run",
            },
            {
                "Item": "Map page",
                "Originally planned": "India district map",
                "What was done": "Not built (future work)",
            },
            {
                "Item": "Dashboard",
                "Originally planned": "8 pages",
                "What was done": "5 pages",
            },
        ]
    )
    st.dataframe(df_planned, use_container_width=True, hide_index=True)
    st.markdown("The scope was narrowed deliberately to methods that the available data can support.")

    st.markdown("#### Why some planned methods were not used")
    st.markdown(
        """
        - DEA and SFA estimate efficiency from school inputs (resources) and outputs. Without district-level school input data they cannot be applied.
        - Moran's I requires district boundary files matched to the 683 analysed districts. This is a separate matching task and is listed as future work. Residuals do cluster by state (for example Punjab and Rajasthan above prediction, Tamil Nadu and Meghalaya below), but no spatial test was performed.
        - An India district map needs the same boundary matching and is listed as the next step.
        """
    )

    st.markdown("---")

    # 1. Pipeline Steps
    st.subheader("1. Data Pipeline Steps")
    st.markdown(
        """
        - **Step 1: NAS 2021 Report Cards Parsed**: Extracted standardized learning achievement scores and survey standard errors
          across Class 3 (Language, Math, EVS), Class 5 (Language, Math, EVS), Class 8 (Language, Math, Science, Social Science),
          and Class 10 (Language, Math, Science, Social Science, English) from official NCERT NAS 2021 district report cards (719 districts).
        - **Step 2: NFHS-5 Indicators Merged**: Processed 706 district fact sheets from the National Family Health Survey (NFHS-5, 2019–21),
          standardizing state names and matching district names using exact matching and manually verified string reconciliation.
        - **Step 3: Conflict Check & Clean Analysis Table**: Verified 1-to-1 uniqueness (0 duplicates or cross-district collisions)
          to create the final matched analytical dataset of **683 districts** across **36 States and Union Territories**.
        """
    )

    st.markdown("---")

    # 2. Dropped Districts
    st.subheader("2. Dropped Districts (36 Districts)")
    st.markdown(
        "A total of **36 NAS districts** were excluded from the analysis table because they could not be matched "
        "cleanly 1-to-1 to an NFHS-5 revenue district."
    )

    dropped_data = [
        ("Arunachal Pradesh", "Capital Complex Itanagar", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Arunachal Pradesh", "Kamle", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Arunachal Pradesh", "Lepa Rada", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Arunachal Pradesh", "Lower Siang", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Arunachal Pradesh", "Pakke Kessang", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Arunachal Pradesh", "Shi Yomi", "Newly carved district without distinct NFHS-5 survey counterpart"),
        ("Delhi", "North West A", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Delhi", "North West B", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Delhi", "South West A", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Delhi", "South West B", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Delhi", "West A", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Delhi", "West B", "Administrative sub-district division in NAS (A/B bifurcations) not matched 1:1 to NFHS revenue districts"),
        ("Karnataka", "Belagavi", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Belagavi Chikkodi", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Bengaluru U North", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Bengaluru U South", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Tumakuru", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Tumakuru Madhugiri", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Karnataka", "Uttara Kannada Sirsi", "Educational sub-district division in NAS with multiple NAS units per NFHS district"),
        ("Madhya Pradesh", "Niwari", "Carved from Tikamgarh in 2018; surveyed under Tikamgarh in NFHS-5"),
        ("Maharashtra", "Mumbai II", "No confirmed NFHS-5 counterpart; left out pending manual verification."),
        ("Manipur", "Jiribam", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Kakching", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Kamjong", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Kangpokpi", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Noney", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Pherzawl", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Manipur", "Tengnoupal", "Newly created district (post-2016 bifurcation); NFHS-5 only surveyed the 9 parent districts"),
        ("Mizoram", "Saiha", "No confirmed NFHS-5 counterpart; left out pending manual verification."),
        ("Telangana", "Mulugu", "New district established in 2019 without separate NFHS-5 sample"),
        ("Telangana", "Narayanapet", "New district established in 2019 without separate NFHS-5 sample"),
        ("West Bengal", "Alipurduar", "Recent district reorganisations/educational divisions without clean 1:1 match"),
        ("West Bengal", "Jhargram", "Recent district reorganisations/educational divisions without clean 1:1 match"),
        ("West Bengal", "Kalimpong", "Recent district reorganisations/educational divisions without clean 1:1 match"),
        ("West Bengal", "Purba Bardhaman", "No confirmed NFHS-5 counterpart; left out pending manual verification."),
        ("West Bengal", "Siliguri", "Recent district reorganisations/educational divisions without clean 1:1 match"),
    ]

    df_dropped = pd.DataFrame(dropped_data, columns=["State", "District", "Reason for Exclusion"])
    st.dataframe(df_dropped, use_container_width=True, height=350)

    st.markdown("---")

    # 3. Limitations
    st.subheader("3. Analytical & Data Limitations")
    st.markdown(
        """
        - **Separate Surveys**: The National Achievement Survey (NAS 2021) and National Family Health Survey (NFHS-5, 2019–21)
          were conducted under independent sampling designs and timelines. Data are linked at the district aggregate level,
          not at the student or household level.
        - **NFHS Women Aged 15–49**: NFHS maternal schooling and health variables measure all surveyed women aged 15–49 in the
          district, serving as a district-level community proxy rather than the specific biological mothers of tested Class 8 students.
        - **Correlation, Not Causation**: All regression coefficients represent observational associations at the ecological (district)
          level and should not be interpreted as causal relationships.
        - **Positive Early-Marriage Coefficient**: In Class 8 regressions, the coefficient on early marriage (`early_marriage`)
          is slightly positive. This is due to statistical collinearity and demographic overlap with maternal schooling and rural household
          structures, rather than any direct pedagogical benefit of early marriage.
        """
    )
