# Robotics Exposure in European Labour Markets
### Using Latent Semantic Analysis to Extend Webb (2019) to ESCO Occupations

**M2 Big Data — Sorbonne School of Economics, Université Paris 1 Panthéon-Sorbonne**  
---

## Research Question

Can Latent Semantic Analysis (LSA) improve the identification of robotics-prone occupations compared to simple keyword matching, and how does robotics exposure reshape wages and employment in Europe?

---

## Methodology

We extend the task-based framework of Acemoglu & Autor (2011) and the patent-text overlap method of Webb (2019) to European occupational data.

**Stage 1 — Baseline exposure (Webb method)**  
We compute a keyword-based robotics exposure score for each occupation by measuring the overlap between ESCO task descriptions and a vocabulary derived from Webb (2019)'s robotics patent verb-noun pairs.

**Stage 2 — LSA exposure (our contribution)**  
We build a joint TF-IDF matrix from O\*NET and ESCO texts, apply Singular Value Decomposition (100 components), and compute cosine similarity between each ESCO occupation vector and a robotics centroid — the mean LSA vector of the most robotics-exposed O\*NET occupations.

**Stage 3 — Prediction model**  
We train Random Forest and LightGBM regressors on the 100 LSA dimensions to predict the keyword baseline score, evaluated with 5-fold cross-validation.

**Stage 4 — Economic inference**  
We regress LSA exposure scores on Eurostat SES 2022 mean hourly wages using OLS with country fixed effects, across 34 European countries and 9 ISCO occupational groups.

---

## Key Results

| Result | Value |
|--------|-------|
| Correlation: LSA vs keyword baseline | r = 0.86 |
| LightGBM cross-validated R² | 0.915 |
| Wage regression coefficient (LSA) | −0.072*** |
| Countries in regression | 34 |
| ESCO occupations scored | 3,039 |

A 1 SD increase in robotics exposure is associated with a **7.2% decrease in hourly wages**, controlling for country fixed effects (p < 0.001).

The polarisation plot shows an inverted-U relationship between robotics exposure and wage percentile rank: **machine operators (OC8) and craft workers (OC7)** — middle-wage occupations — face the highest robotics exposure, consistent with Acemoglu & Autor (2011).

---

## Repository Structure

```
robotics-exposure-europe/
│
├── notebooks/
│   ├── 01_preprocessing_lsa.ipynb     # Text cleaning, baseline + LSA scores
│   ├── 02_prediction_model.ipynb      # Random Forest & LightGBM, SHAP
│   └── 03_regression_analysis.ipynb   # Country FE regression, polarisation plot
│
├── data/
│   └── README.md                      # Data sources and download instructions
│
└── outputs/
    └── figures/                       # All generated figures
```

---

## Data Sources

| Dataset | Source | Used for |
|---------|--------|---------|
| ESCO v1.2.1 | European Commission | Occupational text (3,039 occupations) |
| O\*NET Task Statements | US Dept. of Labor | Baseline keyword scoring |
| Eurostat SES 2022 (`earn_ses22_14`) | Eurostat | Mean hourly wages by occupation |

---

## References

- Acemoglu, D. & Autor, D. (2011). Skills, tasks and technologies. *Handbook of Labor Economics*, 4, 1043–1171.
- Webb, M. (2019). The impact of artificial intelligence on the labor market. SSRN 3482150.
