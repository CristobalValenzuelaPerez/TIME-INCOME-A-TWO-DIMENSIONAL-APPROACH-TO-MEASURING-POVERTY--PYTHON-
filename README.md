# TIME & INCOME: A TWO-DIMENSIONAL APPROACH TO MEASURING POVERTY

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20H2O-F7931E?style=flat-square)](#)
[![Stats](https://img.shields.io/badge/Statistics-Statsmodels-8CA1AF?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

## Project Objective
Traditional economics often measures poverty strictly through an income lens. However, time—like money—is a finite, critical resource. This project develops a robust mathematical framework to identify the **"Time-Income Trap"**: individuals who lack the monetary resources to buy services on the market, *and* lack the free time to perform unpaid care or domestic work themselves.

Using the **UK Time Use Survey (2014-2015)**, this analytical engine processes raw microdata to engineer a holistic index of resources, deploying a full pipeline of unsupervised clustering, rigorous econometrics, and supervised Machine Learning to predict vulnerability to this multidimensional poverty trap.

<img width="348" height="262.8" alt="graph 1" src="https://github.com/user-attachments/assets/1f217988-12ce-4c75-b7f2-18ad81a4f851" />

## Data Source & Reproducibility
The raw microdata utilised in this project is sourced from the **UK Time Use Survey (2014-2015)**. 
Due to strict data governance and respondent privacy regulations, the raw `.dta` files are not included in this repository. 

Researchers and analysts can request access to the original datasets directly through the official UK Data Service portal:
[UK Data Service - UK Time Use Survey 2014-2015](https://datacatalogue.ukdataservice.ac.uk/studies/study/8128#details)

## Methodology & Machine Learning Pipeline

### 1. Feature Engineering & Data Wrangling (`Pandas`, `NumPy`)
* Synthesised complex 24-hour diary entries into aggregated time-use categories (Personal Care, Employment, Unpaid Work).
* Established **Relative Poverty Thresholds** (set at 60% of the median for both Free Time and Disposable Income).
* Engineered a custom `free_time_income_index` to evaluate the intersectionality of deprivations.

### 2. Unsupervised Learning & Dimensionality Reduction (`H2O`, `Scikit-Learn`, `UMAP`)
* **Clustering (DBSCAN & K-Means):** Explored hidden cultural patterns affecting the gender division of unpaid work.
* **UMAP Visualisation:** Reduced high-dimensional feature spaces into a 2D projection to visually validate cultural time-allocation behaviours.

<img width="348" height="348" alt="graph UMAP Dimensionality Reduction" src="https://github.com/user-attachments/assets/3cd0b21e-bdc7-4a45-9f9a-77fd6cc2d9fd" />

### 3. Statistical Inference (`Statsmodels`)
* **Multinomial Logistic Regression:** Evaluated the structural drivers of multidimensional poverty. Handled complete separation issues (Hauck-Donner effect) by applying a **General-to-Specific** modelling approach to isolate highly significant variables.
* **Diagnostics:** Conducted rigorous residual analysis (KDE and Q-Q plots) and multicollinearity checks (VIF) to ensure causal robustness.

### 4. Predictive Modelling (`Imbalanced-Learn`, `Scikit-Learn`)
* **Data Balancing:** Addressed the highly imbalanced nature of the target variable (only ~2.8% of the sample in the "Time & Income Poor" trap) using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Classification Models:** Trained rule-based and probabilistic classifiers to predict poverty trap vulnerability using the refined structural feature set. 
  * *Naïve Bayes Accuracy:* 59.63%
  * *Decision Tree (Rule Induction) Accuracy:* **77.67%** (achieving robust F1-Scores of 0.80 across balanced classes without overfitting).

## Key Findings: The Anatomy of the Poverty Trap
The econometric inference revealed that simultaneous time and income poverty is not random, but deeply structural. The refined model identified highly significant predictors (p < 0.05):

1. **The Gender & Care Burden:** Being female and having infants in the household are the strongest demographic drivers pushing individuals into extreme time poverty.
2. **Material Assets:** The lack of time-saving appliances (e.g., Dishwashers) and the lack of personal transport (Vehicles) exponentially increase the likelihood of falling into the trap.
3. **Household Structure:** Single parents with young children face the highest intersectional vulnerability, highlighting where public policy interventions (like subsidised childcare) would be most effective.

<img width="626.43" height="262.8" alt="graph child04_HH&#39;, &#39;Infants in Household" src="https://github.com/user-attachments/assets/60db376b-7155-4892-bdf7-c5dcfb19d9cb" />

<img width="626.43" height="262.8" alt="graph VehOwn&#39;, &#39;Vehicle in Household" src="https://github.com/user-attachments/assets/81ec1b3e-e89b-4a26-9d4d-d5404cf0643a" />


<p float="left">
  <img src="images/graph%20child04_HH',%20'Infants%20in%20Household.png" width="49%" />
  <img src="images/graph%20VehOwn',%20'Vehicle%20in%20Household.png" width="49%" />
</p>

## Repository Structure
* `multi_dimensional_poverty.ipynb`: The main Jupyter Notebook containing the full end-to-end code.
* `requirements.txt`: List of dependencies required to reproduce the environment.
* `/images/`: Directory containing EDA visualisations and UMAP plots used in this documentation.

---
*Developed by Cristobal Valenzuela-Perez*
