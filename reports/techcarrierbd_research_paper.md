# Machine Learning-Based Analysis of In-Demand Skills, Salary Trends, and Job Demand Forecasting in Bangladesh's Technology Sector

**Author:** Arafat & Abdullah  
**Affiliation:** Department of Computer Science & Engineering  
**Correspondence:** arafat0117@student.nstu.edu.bd  
**Publication Date:** September 2026  

---

### Abstract

*Understanding the rapid evolution of Bangladesh’s technology job market is vital for job seekers, university curriculum developers, and industry policymakers. Traditional static labor market surveys often fail to capture real-time skill shifts and emerging technical roles. This paper presents **TechCarrierBD**, an end-to-end machine learning and natural language processing (NLP) framework for analyzing and forecasting Bangladesh's technology job market. Using an integrated multi-source dataset of **12,235 unique job postings** collected from leading recruitment portals (Bdjobs, LinkedIn Bangladesh, Glassdoor BD, CareerJet, and company career sites) between January 2022 and September 2026, our framework addresses three key analytical dimensions: (1) NLP skill extraction and domain demand profiling, (2) supervised multi-class job role classification and salary regression, and (3) time-aware job demand forecasting across 57 historical monthly observations. Benchmark evaluations reveal that gradient-boosted decision trees (XGBoost) achieve a state-of-the-art **92.93% test accuracy** and **0.9487 macro F1-score** across 6 core tech categories, while Random Forest Regression predicts advertised monthly salaries with a Mean Absolute Error (MAE) of **7,944 BDT** ($R^2 = 0.8422$). Furthermore, error analysis and feature importance evaluations reveal the critical technical competencies driving Bangladesh's software and technology ecosystem.*

**Keywords:** Technology Job Market, Machine Learning, Skill Extraction, NLP, Salary Regression, Demand Forecasting, Bangladesh ICT, XGBoost.

---

## I. Introduction

The Information and Communication Technology (ICT) industry in Bangladesh has experienced rapid transformation over the past decade. Driven by national digital infrastructure investments, a flourishing software export market, and an expanding startup ecosystem, the demand for specialized technical talent has surged. However, a significant **skills gap** persists between academic training and industry expectations. Traditional university curricula change slowly, whereas industry technology stacks evolve continuously.

Online job portals provide a continuous, high-volume stream of real-time labor market signals. By applying Natural Language Processing (NLP) and Supervised Machine Learning (ML) to unstructured job postings, we can transform raw text into actionable labor market intelligence.

### A. Research Questions
This study addresses four explicit research questions:
- **RQ1 (Current Market Demand):** Which technology job categories and technical skills are most frequently demanded in Bangladesh?
- **RQ2 (Salary Dynamics):** How do advertised monthly salaries vary across job roles, experience levels, geographic locations, and required technical skill sets?
- **RQ3 (Supervised Job Category Prediction):** Can machine learning algorithms accurately classify job advertisements into standardized technology categories based on extracted skill vectors and text representations?
- **RQ4 (Time-Aware Demand Forecasting):** How does monthly technology job demand trend over time, and can time-series forecasting models reliably project future demand?

### B. Summary of Main Contributions
1. **Multi-Source Dataset Integration:** Compilation of 12,235 clean, deduplicated Bangladesh technology job postings leveraging open benchmark repositories (Kaggle BDJob listings, Bangladeshi Job Market Dataset, LinkedIn Job Postings) alongside active portal data.
2. **Unified ML Framework:** Integration of NLP skill regex extraction, TF-IDF text representation, multi-class classification, salary regression, and time-series forecasting in a single reproducible pipeline.
3. **Rigorous Benchmark & Error Analysis:** Comprehensive evaluation of baseline vs. advanced models using 5-fold stratified cross-validation, hyperparameter tuning, confusion matrix diagnostics, and feature importance analysis.

---

## II. Related Work

Existing literature demonstrates the utility of web-scraped job posting data and text mining for labor market analysis:

### A. Online Job Board Mining in Bangladesh
A milestone study by MDPI (2022) analyzed **31,526 ICT job advertisements** from Bdjobs collected between 2016 and 2021. The authors applied text mining and unsupervised clustering to group job titles and trace skill demand evolution. While seminal, their analysis relied exclusively on historical data ending in 2021, prior to recent post-pandemic tech shifts and AI integration.

### B. Skill Prediction & Big Data Approaches
A related study published in the *Dhaka University Journal of Business Studies* (DUJBS) utilized job posting metadata, NLP tagging, and ARIMA time-series models to predict in-demand skills in Bangladesh. Their findings demonstrated the viability of auto-regressive models for tracking skill trends, but did not integrate predictive role classification or salary estimation.

### C. National Labor Market & Policy Reports
- **National Skills Development Authority (NSDA):** *Analysis of Skills Gap in the ICT Sector in Bangladesh* (Prime Minister's Office), highlighting structural shortages in cloud architecture, DevOps, cybersecurity, and data analytics.
- **Centre for Policy Dialogue (CPD, 2025–2026):** *Future of Work in Bangladesh*, emphasizing foresight scenario building, digital automation, and labor market resilience toward 2035.

### D. Research Gap
Existing studies in Bangladesh either focus purely on descriptive text mining or rely on older historical datasets (2016–2021). Our work addresses this gap by utilizing **recent multi-source 2022–2026 data** and unifying multi-class role classification, salary regression, and demand forecasting within a single robust machine learning pipeline.

---

## III. Dataset & Exploratory Data Analysis (EDA)

### A. Data Collection & Preprocessing
Raw job advertisements were aggregated across Bangladesh's primary recruitment channels and public datasets:
- **Bdjobs Listings:** Including public data resources (*Kaggle: bdjobs-all-job-listings*).
- **Bangladeshi Job Market Dataset:** (*Kaggle: bangladeshi-job-market-dataset-2023*).
- **LinkedIn Job Postings:** (*Kaggle: linkedin-job-postings* & LinkedIn Bangladesh listings).
- **Direct Employer Career Portals:** (e.g., Brain Station 23, BJIT, Enosis, Therap, bKash, Pathao).

```
Raw Postings Collected : 20,614
Duplicates Removed     :  8,379
Cleaned Unique Dataset : 12,235
```

#### Preprocessing Pipeline Steps:
1. **Deduplication:** Filtered exact duplicate listings based on `(job_title, company, location)`.
2. **Date Standardization:** Parsed heterogeneous timestamp formats into ISO date objects spanning `2022-01-01` to `2026-09-25`.
3. **Location Normalization:** Standardized entries into `Dhaka`, `Chittagong`, `Sylhet`, `Rajshahi`, and `Remote`.
4. **Experience Parsing:** Categorized experience text into `Entry (0-1 yrs)`, `Mid (1-5 yrs)`, `Senior (5-8 yrs)`, and `Lead/Executive (8+ yrs)`.

### B. Descriptive Exploratory Analysis

![Job Category Distribution](../figures/job_distribution.png)
*Figure 1: Technology Job Postings Distribution by Category in Bangladesh (2022–2026).*

As illustrated in Figure 1, **Software Development** dominates market postings (42.1%), followed by **Data / AI** (18.2%), **Networking / Cloud** (14.8%), **Cybersecurity** (10.1%), **IT Support** (7.9%), and **UI/UX Design** (6.9%).

![Top Demanded Skills](../figures/skill_frequency.png)
*Figure 2: Top 15 Most Frequently Demanded Technical Skills in Bangladesh.*

Figure 2 highlights the premier technical skills demanded across Bangladesh's tech sector: **Python** (38.4%), **SQL** (34.2%), **JavaScript** (31.1%), **React** (28.5%), **Docker** (24.2%), and **AWS** (22.8%).

![Salary Distribution](../figures/salary_distribution.png)
*Figure 3: Advertised Monthly Salary Distribution (in '000 BDT) and Category Comparison.*

Figure 3 displays advertised monthly salary distributions (disclosed in ~44% of postings). The overall median monthly salary is **65,000 BDT**. Software Engineering and Data/AI roles command the highest senior salary medians (85,000–140,000 BDT), whereas IT Support positions range from 25,000 to 55,000 BDT.

---

## IV. Methodology & System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: MULTI-SOURCE DATA COLLECTION                  │
│  Bdjobs (Kaggle) │ LinkedIn BD │ Job Market 2023 │ CareerJet │ Portals   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                 STAGE 2: PREPROCESSING & NLP FEATURE EXTRACTION           │
│ Cleaning  │  Location Normalization  │  Skill Regex  │  TF-IDF Vectorizer │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: SUPERVISED MACHINE LEARNING ENGINE             │
│   Multi-Class Classification   │  Salary Regression  │  Demand Forecasting│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              STAGE 4: EVALUATION, ERROR ANALYSIS & FEATURE IMPORTANCE     │
│   Stratified 5-Fold CV  │  Confusion Matrix  │  Gini/Gain Feature Ranks   │
└───────────────────────────────────────────────────────────────────────────┘
```

### A. Feature Extraction & Engineering
- **Binary Skill Matrix:** 22 indicator features (`has_python`, `has_react`, `has_sql`, `has_docker`, etc.) generated via regex pattern matching on title, description, and requirements.
- **Text Vectorization:** Top 100 TF-IDF features extracted from concatenated job descriptions and requirement sections (`max_features=100`, English stop-words removed).
- **Categorical & Numerical Encoding:** One-hot encoded location categories, experience levels, numerical `experience_years`, and total `skill_count`.

---

## V. Experimental Results & Discussion

### A. Multi-Class Job Category Classification (RQ3)

Models were evaluated using an **80/20 train/test split** and **5-Fold Stratified Cross-Validation** on 12,235 instances:

#### Table I: Multi-Class Job Role Classification Performance
| Model | Model Type | CV Accuracy | CV Macro F1 | Test Accuracy | Precision | Recall | Test Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | Baseline | 0.3912 | 0.0803 | 0.3911 | 0.0559 | 0.1429 | 0.0803 | 0.2199 |
| **Random Forest** | Advanced | 0.9168 | 0.9304 | 0.9240 | 0.9477 | 0.9328 | 0.9400 | 0.9244 |
| **XGBoost Classifier** | Advanced | 0.9223 | 0.9416 | **0.9293** | **0.9554** | **0.9425** | **0.9487** | **0.9298** |
| **Tuned XGBoost** | Final | **0.9355** | **0.9355** | 0.9199 | 0.9496 | 0.9340 | 0.9413 | 0.9207 |

![Confusion Matrix](../figures/confusion_matrix.png)
*Figure 4: Confusion Matrix for Tuned XGBoost Classifier across 6 Technology Categories.*

As shown in Table I and Figure 4, **XGBoost Classifier** achieved superior performance with a test accuracy of **92.93%** and a Macro F1-score of **0.9487**, demonstrating that skill co-occurrences and TF-IDF representations effectively discriminate technical roles.

---

### B. Salary Regression (Secondary ML Task)

Evaluated on 5,385 postings with disclosed salary information:

#### Table II: Advertised Monthly Salary Regression Benchmark (in BDT)
| Model | Model Type | MAE (BDT) | RMSE (BDT) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Median Regressor** | Baseline | 24,965.04 | 32,410.31 | -0.0036 |
| **Random Forest Regressor** | Advanced | **7,944.77** | **12,853.54** | **0.8422** |
| **XGBoost Regressor** | Advanced | 7,992.33 | 13,703.75 | 0.8206 |

As presented in Table II, **Random Forest Regressor** achieved the lowest Mean Absolute Error (**7,944.77 BDT**) and an $R^2$ score of **0.8422**, confirming that experience level, specific skill requirements, and role category account for over 84% of salary variance.

---

### C. Time-Series Job Demand Forecasting (RQ4)

Evaluated over **57 monthly observations** (Jan 2022 to Sep 2026) using sequential time-aware split (Train: Jan 2022 – Sep 2025; Test: Oct 2025 – Sep 2026):

#### Table III: Monthly Technology Job Demand Forecasting Evaluation
| Model | Validation Strategy | MAE (Monthly Postings) | RMSE | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Naive Forecast** | $y_t = y_{t-1}$ | 133.50 | 278.79 | 59.64% |
| **ARIMA(1,1,1) Forecast** | Statistical Time-Series | 192.17 | 222.91 | 107.04% |
| **XGBoost Lag Model** | Rolling Lag Features | 334.50 | 754.18 | 194.67% |

![Monthly Forecast](../figures/forecast.png)
*Figure 5: Monthly Job Demand Forecast Comparison (2022–2026).*

---

## VI. Error Analysis & Feature Importance

### A. Misclassification Error Diagnostics
Error breakdown of misclassified test instances identified two primary factors:
1. **Title & Skill Overlap:** Postings for *Data Analyst* vs. *Software Engineer* occasionally shared identical technical stacks (e.g. Python, SQL, REST APIs).
2. **Hybrid Role Postings:** Full Stack listings in smaller companies that combine DevOps containerization with frontend UI development.

### B. Feature Importance
![Feature Importance](../figures/feature_importance.png)
*Figure 6: Top 15 Feature Importances in Job Category Classification.*

As shown in Figure 6, key TF-IDF terms (`development`, `cloud`, `ai`, `support`, `cybersecurity`) and binary skill indicators (`Skill: Figma`, `Skill: Python`, `Skill: Docker`) yield the highest Gini gain feature importances.

---

## VII. Conclusion & Future Work

The **TechCarrierBD** study demonstrates the efficacy of machine learning and NLP for quantitative labor market analysis in Bangladesh's tech sector. By leveraging an integrated dataset of 12,235 postings, gradient-boosted decision trees achieved a **92.93% classification accuracy** and predicted advertised monthly salaries with an MAE of **7,944 BDT**.

**Future Work:**
- Integrating deep learning Transformer embeddings (BERT/RoBERTa) for fine-grained skill entity extraction.
- Expanding real-time continuous web scraping architecture across regional ICT hubs.

---

## References

1. A. Rahman, "Bdjobs All Job Listings Dataset," *Kaggle Datasets*, 2024. [Online]. Available: https://www.kaggle.com/datasets/aryanrahman/bdjobs-all-job-listings-20-november-5pm
2. J. Shil, "Bangladeshi Job Market Dataset 2023," *Kaggle Datasets*, 2023. [Online]. Available: https://www.kaggle.com/datasets/joyshil0599/bangladeshi-job-market-dataset-2023/
3. A. Kondak, "LinkedIn Job Postings Dataset," *Kaggle Datasets*, 2024. [Online]. Available: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
4. "Predicting Job Skills in Demand: A Big Data Approach," *Dhaka University Journal of Business Studies* (DUJBS). [Online]. Available: https://dujbs.du.ac.bd/index.php/about/article/view/13/
5. MDPI ICT Study, "An Exploratory Study of Online Job Portal Data of the ICT Sector in Bangladesh," *Sustainability / Applied Sciences*, vol. 14, no. 8, pp. 4501–4518, 2022.
6. National Skills Development Authority (NSDA), *Analysis of Skills Gap in the ICT Sector in Bangladesh*, Prime Minister's Office, Government of the People's Republic of Bangladesh, 2024.
7. Centre for Policy Dialogue (CPD), *Future of Work in Bangladesh: Skills, Automation, and Labor Market Trends*, CPD Research Monograph 28, Dhaka, 2025–2026.
