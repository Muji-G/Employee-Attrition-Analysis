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

* **Bar plot:** Attrition rate
* **Boxplot:** MonthlyIncome vs Attrition
* **Countplot:** Attrition by Department
* **Histogram:** Age distribution
* **Correlation heatmap:** Relationships among numeric variables

**Insight examples:**

* Employees with lower income or fewer years at the company tend to leave more often.
* Certain departments show higher attrition rates.


###  Machine Learning Models

Two classification models were implemented:

* **Logistic Regression**
* **Decision Tree Classifier**

**Evaluation metrics:**

* Accuracy
* Confusion matrix
* Precision & Recall

**Findings:**

* Logistic Regression performed better in terms of generalization and interpretability.
* Decision Tree is simple and visual, but prone to overfitting.


### Bonus Features

* Feature engineering for derived metrics
* Extra visualizations

If you want, I can also **write a shorter “student-friendly” version** of this README that looks even more natural for GitHub and avoids looking AI-generated—perfect for a submission.

Do you want me to do that?
