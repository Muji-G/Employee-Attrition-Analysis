# Employee Attrition Analysis

## Project Overview

This project analyzes employee attrition using the IBM HR Analytics dataset. The goal is to understand why employees leave the company, identify key factors driving attrition, and build machine learning models to predict which employees are likely to leave.

By combining data cleaning, exploratory data analysis (EDA), and predictive modeling, this project provides actionable insights for HR management and helps improve employee retention strategies.

## Dataset

IBM HR Analytics Employee Attrition Dataset

* **Rows:** ~1,470 employees
* **Columns:** Numeric and categorical features including age, job role, income, years at company, and more
* **Target variable:** `Attrition` (Yes / No)

The dataset is clean, well-documented, and contains meaningful features for business analysis.


## Methodology

###  Data Cleaning & Processing

* Checked for missing values (dataset is mostly clean)
* Encoded categorical variables (LabelEncoder / One-Hot Encoding)
* Removed irrelevant columns (EmployeeNumber, Over18, etc.)
* Created 1–2 derived features for extra insight:

  * `YearsAtCompany / Age`
  * `MonthlyIncome per JobLevel`

**Goal:** Prepare a clean dataset suitable for modeling while maintaining business meaning.



###  Exploratory Data Analysis (EDA)

Key visualizations include:

* **Boxplot:** Distance from home vs 
* **Histogram:** Many ))
* **Correlation heatmap:** Relationships among numeric variables

**Insights:**

* The workers with low JobLevel, MonthlyIncome, YearAtCompany, and TotalWorkingYears are more likely to quit their jobs.
* As it seems EnvironmentSatisfaction, JobSatisfaction, and RelationshipSatisfaction features don't have a big impact on the determination of Attrition of employees.

###  Machine Learning Models

| Model               | Accuracy | Precision | Recall | Summary                                                                                                        |
| ------------------- | -------- | --------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | 88.7%    | 76.9%     | 42.3%  | Best overall balance. Correctly predicts most cases, with good precision, but may miss some actual attritions. |
| Decision Tree       | 75.1%    | 28.1%     | 35.2%  | Low performance. Many false positives and misses actual attritions. Not reliable.                              |
| Random Forest       | 82.3%    | 36.0%     | 12.7%  | Poor recall. Overly cautious and misses most actual attritions. Needs tuning.                                  |

**Findings:**

* Logistic Regression is the best model overall in terms of balanced performance.
* Precision vs Recall trade-off: If the goal is to catch as many positives as possible (e.g., churners), Logistic Regression might need tuning, such as adjusting the classification threshold or using resampling to balance classes.
* Decision Tree and Random Forest are underperforming, likely due to overfitting, class imbalance, or insufficient hyperparameter tuning

** Recommendation:**

* Stick with Logistic Regression for now, but try improving recall using:
* Threshold adjustment
* Ensemble methods or tuned hyperparameters
* Avoid using Decision Tree or Random Forest without tuning, as they are less reliable for this dataset.

### Bonus Features

* Feature engineering for derived metrics
* Interactive visualizations
* Recommendations

Sorry for the delay
May the 4th be with you))
