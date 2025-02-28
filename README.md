# Data Science Salary Analysis

## 📌 Overview
This project analyzes salary trends in data science roles across different factors such as experience level, job title, company size, location, and employment type. The goal is to extract meaningful insights to help job seekers, professionals, and employers make data-driven decisions.

## 🚀 Project Overview

- **Data Source**: Salary dataset for data science professionals  
- **Objective**: Identify trends, salary distributions, and key factors influencing salaries  
- **Tech Stack**: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, XGBoost, Ridge & Lasso Regression  

---

## 📖 Table of Contents

1. [Import Necessary Libraries 📚](#import-necessary-libraries)
2. [Load and Explore the Dataset 🔍](#load-and-explore-the-dataset)
   - 2.1 [Basic Information](#basic-information)
   - 2.2 [Missing Values Analysis](#missing-values-analysis)
   - 2.3 [Summary Statistics](#summary-statistics)
3. [Exploratory Data Analysis (EDA) 📊](#exploratory-data-analysis-eda)
   - 3.1 [Univariate Analysis](#univariate-analysis)
       - 3.1.1 [Salary Distribution](#salary-distribution)
       - 3.1.2 [Work Year Distribution](#work-year-distribution)
       - 3.1.3 [Experience Level Breakdown](#experience-level-breakdown)
       - 3.1.4 [Employment Type Overview](#employment-type-overview)
       - 3.1.5 [Job Title Distribution](#job-title-distribution)
       - 3.1.6 [Employee Residence Distribution](#employee-residence-distribution)
       - 3.1.7 [Company Location Breakdown](#company-location-breakdown)
       - 3.1.8 [Company Size Analysis](#company-size-analysis)
       - 3.1.9 [Key Takeaways from Univariate Analysis](#key-takeaways-from-univariate-analysis)
   - 3.2 [Multivariate Analysis](#multivariate-analysis)
       - 3.2.1 [Work Year vs Salary](#work-year-vs-salary)
       - 3.2.2 [Experience Level vs Salary](#experience-level-vs-salary)
       - 3.2.3 [Employment Type vs Salary](#employment-type-vs-salary)
       - 3.2.4 [Job Title vs Salary](#job-title-vs-salary)
       - 3.2.5 [Employee Residence vs Salary](#employee-residence-vs-salary)
       - 3.2.6 [Company Size vs Salary](#company-size-vs-salary)
       - 3.2.7 [Company Location vs Salary](#company-location-vs-salary)
   - 3.3 [Bonus: Job Title Recommendation for Entry-level Candidates](#bonus-job-title-recommendation-for-entry-level-candidates)
4. [Feature Engineering ⚙️](#feature-engineering)
   - 4.1 [Encoding Categorical Variables](#encoding-categorical-variables)
   - 4.2 [Creating New Features](#creating-new-features)
   - 4.3 [Handling Skewed Data](#handling-skewed-data)
5. [Modeling 🔎](#modeling)
   - 5.1 [Preprocessing Data for Modeling](#preprocessing-data-for-modeling)
   - 5.2 [Splitting Data into Train and Test Sets](#splitting-data-into-train-and-test-sets)
   - 5.3 [Model Selection and Evaluation](#model-selection-and-evaluation)
       - 5.3.1 [Linear Regression](#linear-regression)
       - 5.3.2 [Ridge & Lasso Regression](#ridge-lasso-regression)
       - 5.3.3 [Random Forest Regression](#random-forest-regression)
       - 5.3.4 [XGBoost Regression](#xgboost-regression)
   - 5.4 [Hyperparameter Tuning](#hyperparameter-tuning)
   - 5.5 [Feature Importance Analysis](#feature-importance-analysis)
   - 5.6 [Comparing Model Performance](#comparing-model-performance)
   - 5.7 [Final Model Selection and Insights](#final-model-selection-and-insights)
6. [Conclusion & Business Insights 💡](#conclusion-business-insights)
   - 6.1 [Key Findings](#key-findings)
   - 6.2 [Salary Trends & Drivers](#salary-trends-drivers)
   - 6.3 [Recommendations for Job Seekers & Employers](#recommendations-for-job-seekers-employers)
   - 6.4 [Limitations & Future Improvements](#limitations-future-improvements)



---

## 📊 Data Preprocessing

- **Missing Values**: Handled appropriately
- **Categorical Encoding**:
  - Label Encoding (For Ordinal Categories)
  - One-Hot Encoding (For Non-Ordinal Categories)
- **Feature Scaling**: Standardization applied where necessary
- **Train-Test Split**: 80% training, 20% testing

---

## 🔎 Feature Selection

The dataset contains various features related to salaries. After careful selection, the key influencing factors used in the model are:

- **work_year**  
- **experience_level**  
- **job_title**  
- **salary_in_usd**  
- **remote_ratio**  
- **company_size**  
- **company_location**  
- **employment_type**  

---

## 📈 Salary Trends and Insights

### **Salary Trends Over Time**
- Minimal salary change in 2020-2021  
- Fluctuations observed in 2021-2022  

### **Impact of Experience**
- More experience leads to higher salaries  
- Leadership and niche expertise roles earn significantly higher  

### **Employment Types**
- **Full-Time** employees have the highest salaries  
- Contractors earn well but with high variation  
- Freelancers and part-timers earn the least  

### **Top Job Titles and Salaries**
- **Highest Paying Roles**: Data Science Manager, ML Scientist, Data Architect  
- **Technical Roles**: Data Scientist, Data Engineer, ML Engineer earn moderately well  
- **Entry-Level**: Data Analyst, Big Data Engineer  

### **Company Size Impact**
- Medium-sized companies see better salary growth than large firms  
- Large companies provide stable salaries  

### **Currency & Location-Based Salaries**
- **Highest Salaries**: USD, ILS, GBP, CHF  
- **Top Locations**: US, Japan, Canada  
- **Lowest Salaries**: India, Greece, Spain  

### **Salary Distribution**
- Right-skewed salary distribution (more low-mid salaries)  
- Most professionals earn within a specific salary range  

---

## 🏆 Model Performance

| Model                 | MAE    | MSE     | RMSE   | R² Score |
|----------------------|--------|---------|--------|---------|
| **Linear Regression**  | 54.06  | 5361.74 | 73.22  | 0.521   |
| **Ridge Regression**   | 49.70  | 4555.71 | 67.49  | 0.593   |
| **Lasso Regression**   | 49.83  | 4602.11 | 67.83  | 0.589   |
| **Random Forest**      | 54.28  | 5603.64 | 74.85  | 0.499   |
| **XGBoost**           | 55.23  | 5966.74 | 77.24  | 0.467   |

### **Best Performing Model**
🏆 **Ridge Regression**  
- **Lowest MAE** (49.70) → Most accurate predictions on average  
- **Lowest MSE & RMSE** → Smaller overall errors  
- **Highest R² Score (0.5933)** → Explains ~59.3% of salary variance  

---

## 📌 Recommendations for Data Science Professionals

1. **Gain More Experience** – Higher levels lead to significantly better salaries.  
2. **Specialize in Leadership/Niche Roles** – Data Science Manager, ML Scientist earn the most.  
3. **Consider Medium-Sized Companies** – They offer better salary growth than large firms.  
4. **Location Matters** – The US, Japan, and Canada offer the highest-paying jobs.  
5. **Freelancing/Contracting** – Experienced contractors can earn up to **$416,000 USD**.  
6. **Technical Skills** – Data Science, ML, and Engineering roles provide steady salaries.  

---
### Contributing

I welcome contributions! If you have ideas for improving this project:

1. Fork the repository.
2. Create a new branch (feature-branch).
3. Commit your changes.
4. Submit a Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contact

For queries or collaborations, reach out via:
- Email: himunagapure114@gmail.com
- LinkedIn: https://www.linkedin.com/in/himanshunagapure/
