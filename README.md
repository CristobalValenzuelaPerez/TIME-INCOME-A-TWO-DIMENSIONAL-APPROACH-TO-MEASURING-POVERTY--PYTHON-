# TIME & INCOME: A TWO-DIMENSIONAL APPROACH MEASURING POVERTY (Python)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20H2O-F7931E?style=flat-square)](#)
[![Stats](https://img.shields.io/badge/Statistics-Statsmodels-8CA1AF?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)


## Project Objective
Traditional economics often measures poverty strictly through an income lens. However, time—like money—is a finite resource. This project develops a robust mathematical framework to identify the **"Time-Income Trap"**: individuals who lack the monetary resources to buy services on the market, *and* lack the free time to perform unpaid care or domestic work themselves.

Using the **UK Time Use Survey (2014-2015)**, this Python engine processes over 335 raw variables to engineer a holistic index of resources, deploying both unsupervised and supervised Machine Learning algorithms to predict vulnerability to this two-dimensional poverty trap.

## Data Source & Reproducibility
The raw microdata utilized in this project is sourced from the **UK Time Use Survey (2014-2015)**. 
Due to strict data governance and respondent privacy regulations, the raw `.dta` files are not included in this repository. 

Researchers and analysts can request access to the original datasets directly through the official UK Data Service portal:
[UK Data Service - UK Time Use Survey 2014-2015](https://datacatalogue.ukdataservice.ac.uk/studies/study/8128#details)

## Methodology & Machine Learning Pipeline

### 1. Feature Engineering & Data Wrangling (`Pandas`, `NumPy`)
* Synthesized complex 24-hour diary entries into aggregated time-use categories (Personal Care, Employment, Unpaid Work, Commute).
* Established **Relative Poverty Thresholds** (set at 60% of the median for both Free Time and Disposable Income).
* Engineered a custom `free_time_income_index` to evaluate the intersectionality of deprivations.

### 2. Statistical Inference (`Statsmodels`)
* **Ordinary Least Squares (OLS):** Deployed to measure the impact of socio-demographic features on the time-income index, using Variance Inflation Factor (VIF) to handle multicollinearity.
* **Multinomial Logistic Regression:** Identified key determinants that significantly increase the risk of falling into simultaneous time and income poverty (e.g., household structure, presence of infants, and ownership of time-saving appliances like tumble dryers).

### 3. Unsupervised Learning & Dimensionality Reduction (`H2O`, `Scikit-Learn`, `UMAP`)
* **Clustering (DBSCAN & K-Means):** Explored hidden cultural patterns affecting the gender division of unpaid work.
* **UMAP Visualization:** Reduced high-dimensional clustering outputs into a 2D space for visual validation of ethnic and cultural time-allocation behaviours.

### 4. Predictive Modelling (`Imbalanced-Learn`, `Scikit-Learn`)
* **Data Balancing:** Addressed the highly imbalanced nature of the target variable (only ~2.8% of the sample was in the "Time & Income Poor" category) using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Classification Models:** Trained and validated models to predict the likelihood of an individual falling into the poverty trap. 
  * *Naïve Bayes:* 95.58% Accuracy
  * *Decision Trees / Rule Induction:* **97.28% Accuracy**

## Key Finding: The Poverty Gap
While identifying the poor is critical, calculating the **Poverty Gap** provides actionable intelligence for public policy. 
The algorithm revealed that individuals trapped in simultaneous time and income poverty would need an average of **£525.70 extra per month** to escape income deprivation. However, viewed from a time-need perspective, they only require an additional **6.35 hours of free time per week** to escape the trap. This highlights how targeted state interventions (like subsidised childcare or transport) can be highly cost-effective alternatives to direct monetary subsidies.

## Repository Structure
* `multi_dimensional_poverty.ipynb`: The main Jupyter Notebook containing the full end-to-end code.
* `requirements.txt`: List of dependencies (h2o, umap-learn, imblearn, etc.)
* `/images/`: Directory containing UMAP and correlation matrix plots.

---
*Developed by [Tu Nombre/Perfil de GitHub]*
