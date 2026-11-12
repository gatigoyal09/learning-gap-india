# The Learning Gap in India: District Learning Outcomes and Household Conditions

This study investigates the relationship between district-level student learning outcomes and household socio-economic characteristics across India by reconciling National Achievement Survey (NAS 2021) scores with National Family Health Survey (NFHS-5) district metrics. Fifty-five percent of the total variance in Class 8 district average test scores lies between states, with four NFHS household predictors—maternal schooling, household basic amenities, early marriage, and child stunting—explaining an R² of 0.08 alone in an OLS model and 0.62 when accounting for state fixed effects. Within states, districts with higher proportions of educated mothers score higher on average, though all observed patterns reflect statistical associations rather than causal relationships.

---

## Data

- **NAS 2021 District Report Cards**: Published by the Ministry of Education (NCERT), covering standardized learning achievement scores and sampling standard errors across Class 3, 5, 8, and 10 subjects, parsed directly from official PDF district report cards.
- **NFHS-5 District Factsheets**: Published by the Ministry of Health and Family Welfare (available on [data.gov.in](https://data.gov.in)), providing district-level estimates of female education, household amenities, nutritional status, and demographic indicators collected between 2019 and 2021.

> **Note**: Raw survey files are **NOT** included in this repository and must be downloaded directly from the official sources mentioned above.

---

## Pipeline

The data preparation, matching, and econometric analysis scripts were executed in the following order:

1. `parse_nas.py`: Parses raw NCERT NAS 2021 district report card PDFs into structured test scores across grades and subjects.
2. `create_wide_v2.py`: Reshapes and standardizes NAS subject and grade scores into a district-level wide format table (`nas_district_wide_v2.csv`).
3. `clean_nfhs.py`: Cleans and formats the raw NFHS-5 district factsheet CSV (`nfhs_clean.csv`).
4. `create_nfhs_selected.py`: Extracts selected maternal education, amenity, and child health variables into standardized columns (`nfhs_selected.csv`).
5. `match_nas_nfhs.py`: Reconciles state naming variations and district spelling discrepancies between NAS and NFHS-5, generating candidate match pairs.
6. `build_analysis_table.py`: Performs 1-to-1 conflict validation, joins the NAS and NFHS datasets, and creates the harmonized analytical table of 683 districts (`analysis_table.csv`).
7. `analysis.py`: Estimates Model A (OLS with HC3 robust standard errors) and Model B (State Fixed Effects with state-clustered standard errors), performs variance decomposition, Breusch-Pagan tests, outlier analysis, and robustness checks.
8. `run_post_analysis.py`: Computes Model B residuals, flags top and bottom deciles (above/below predicted performance), and exports the full district dataset (`district_results.csv`).
9. `generate_eda_plots.py`: Generates exploratory data analysis charts and summary plots.

---

## Limitations

- **Separate Surveys**: The National Achievement Survey (NAS 2021) and the National Family Health Survey (NFHS-5, 2019–21) were conducted independently with different sampling frames; data are linked at the district aggregate level, not at the student or household level.
- **NFHS Covers Women Aged 15–49**: NFHS maternal schooling and health indicators reflect all surveyed women aged 15–49 in the district, serving as a community-level proxy rather than the specific mothers of tested Class 8 students.
- **36 NAS Districts Dropped**: A total of 36 NAS districts were excluded from the final analysis table because they could not be matched 1-to-1 to NFHS revenue districts, primarily due to post-2016 district bifurcations in states like Manipur and Arunachal Pradesh, or educational sub-district divisions in Karnataka and Delhi.
- **Observational Data**: All regression coefficients capture cross-sectional ecological correlations and do not imply causal mechanisms. The positive coefficient on early marriage in Class 8 regressions is an artifact of demographic overlap and regional collinearity with maternal schooling and rural structures, rather than a direct educational benefit.

---

## Run Locally

Run locally: pip install -r requirements.txt then streamlit run app/app.py
