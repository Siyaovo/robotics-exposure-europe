# Robotics Exposure Index for European Labour Markets

Improving occupational robotics exposure measurement with **Latent Semantic Analysis (LSA)** — applied to 3,039 European occupations from the ESCO classification.

---

## Overview

Which jobs are most at risk of robotic displacement? The classic answer relies on keyword matching: count how often robotics-related words appear in job descriptions (Webb, 2019). This project proposes a better method using **Latent Semantic Analysis**, which captures *semantic similarity* to robotic tasks rather than exact keyword overlap.

**Key contribution:** LSA extends robotics exposure scores to European occupations (ESCO) that have no direct US equivalent in O\*NET — occupations the keyword method simply cannot score.

---

## Method

### Notebook 01 — Preprocessing & LSA Scoring

1. **Data sources**
   - O\*NET task descriptions (893 US occupations, ~18,000 task statements)
   - ESCO occupations + skills (3,039 occupations, 13,960 skills)

2. **Baseline score** — Webb (2019) keyword matching: fraction of task tokens matching a robotics patent vocabulary (welding, assembly, machining, etc.)

3. **LSA score**
   - Joint TF-IDF matrix fitted on O\*NET + ESCO (20,000 features, bigrams)
   - Truncated SVD reduces to 100 latent dimensions
   - A *robotics centroid* is built from the top-30 O\*NET occupations by keyword score
   - Each ESCO occupation is scored by **cosine similarity** to this centroid

4. **Crosswalk** — semantic mapping from O\*NET to ESCO via LSA similarity, bridging two different occupational classification systems

### Notebook 02 — Prediction Model

- Random Forest and LightGBM trained on 100 LSA dimensions to predict the keyword baseline score
- 5-fold cross-validation: **CV R² = 0.899 (RF) / 0.915 (LightGBM)**
- SHAP analysis identifies which latent semantic dimensions drive robotics exposure

---

## Results

| Model | CV R² | CV MSE |
|---|---|---|
| Random Forest | 0.899 | — |
| LightGBM | 0.915 | — |

High predictive performance confirms that LSA captures the same underlying signal as Webb's keyword method — while extending coverage to 3,039 European occupations, including those with no US equivalent.

---

## Repository Structure

```
robotics-exposure-europe/
├── notebooks/
│   ├── 01_preprocessing_lsa.ipynb   # Data loading, LSA scoring, crosswalk
│   └── 02_prediction_model.ipynb    # Random Forest, LightGBM, SHAP
├── data/
│   ├── raw/                         # O*NET and ESCO source files (not tracked)
│   └── clean/                       # Processed outputs from Notebook 01
├── outputs/
│   └── figures/                     # Charts and SHAP plots
└── README.md
```

---

## Tech Stack

`Python` · `scikit-learn` · `LightGBM` · `SHAP` · `pandas` · `NLTK` · `TF-IDF` · `TruncatedSVD`

---

## Data Sources

- [O\*NET Task Statements](https://www.onetonline.org/) — US occupational task descriptions
- [ESCO v1](https://esco.ec.europa.eu/) — European Skills, Competences, Qualifications and Occupations
- Webb, M. (2019). *The Impact of Artificial Intelligence on the Labor Market.* Working Paper.

---

## Author

**Siyao Zhang** — M2 Sustainable Development Economics, Panthéon-Sorbonne  
[LinkedIn](https://www.linkedin.com/in/siyao-z-707906320/)
