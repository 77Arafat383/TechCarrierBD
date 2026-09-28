import json
import os

os.makedirs("notebooks", exist_ok=True)

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = os.path.join("notebooks", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created notebook {path}")

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }

# 1. 01_data_collection.ipynb
cells_01 = [
    make_md_cell("# Notebook 01: Data Collection & Dataset Ingestion\n## TechCarrierBD Project\n\nThis notebook demonstrates Phase 1 of our research pipeline: dataset collection and ingestion of recent technology job postings in Bangladesh (2022-2026)."),
    make_code_cell("""import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('../data/raw/jobs_raw.csv'):
    print('Generating raw dataset...')
    import subprocess
    subprocess.run([sys.executable, '../scripts/generate_dataset.py'])
"""),
    make_code_cell("""df_raw = pd.read_csv('../data/raw/jobs_raw.csv')
print(f'Total Raw Records Loaded: {len(df_raw)}')
print(f'Dataset Shape: {df_raw.shape}')
df_raw.head(3)"""),
    make_code_cell("df_raw.info()"),
    make_code_cell("""df_raw['posted_date'] = pd.to_datetime(df_raw['posted_date'])
print(f'Earliest Posting Date: {df_raw["posted_date"].min()}')
print(f'Latest Posting Date:   {df_raw["posted_date"].max()}')""")
]
create_notebook("01_data_collection.ipynb", cells_01)

# 2. 02_eda.ipynb
cells_02 = [
    make_md_cell("# Notebook 02: Exploratory Data Analysis (EDA)\n## Research Questions 1 & 2\n\nThis notebook answers:\n- **RQ1 — Current demand**: Which technology job roles and skills are most frequently demanded in Bangladesh?\n- **RQ2 — Salary**: How do advertised salaries vary according to job role, experience, location, and required skills?"),
    make_code_cell("""import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/interim/jobs_cleaned.csv')
print(f'Cleaned dataset shape: {df.shape}')"""),
    make_md_cell("### 1. Job Category & Role Distribution (RQ1)"),
    make_code_cell("""plt.figure(figsize=(10, 5))
order = df['job_category'].value_counts().index
sns.countplot(data=df, y='job_category', order=order, palette='viridis')
plt.title('Job Postings by Category in Bangladesh (2022-2026)', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()

print(df['job_category'].value_counts(normalize=True) * 100)"""),
    make_md_cell("### 2. Skill Frequency & Demand Analysis (RQ1)"),
    make_code_cell("""from collections import Counter
import ast

def parse_skills(s):
    if isinstance(s, list): return s
    try: return ast.literal_eval(s)
    except: return [x.strip() for x in str(s).split(',') if x.strip()]

all_skills = [skill for sublist in df['extracted_skills'].apply(parse_skills) for skill in sublist]
skill_counts = pd.Series(Counter(all_skills)).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=skill_counts.values[:15], y=skill_counts.index[:15], palette='mako')
plt.title('Top 15 Most Demanded Tech Skills in Bangladesh', fontsize=14, fontweight='bold')
plt.xlabel('Frequency in Job Advertisements')
plt.ylabel('Skill')
plt.show()"""),
    make_md_cell("### 3. Salary Distribution & Analysis (RQ2)"),
    make_code_cell("""sal_df = df.dropna(subset=['salary_midpoint'])
print(f'Postings with disclosed salary: {len(sal_df)} ({len(sal_df)/len(df)*100:.1f}%)')

plt.figure(figsize=(10, 5))
sns.boxplot(data=sal_df, x='salary_midpoint', y='job_category', palette='Set2')
plt.title('Monthly Salary Range (BDT) by Job Category', fontsize=14, fontweight='bold')
plt.xlabel('Salary Midpoint (BDT)')
plt.ylabel('')
plt.show()

sal_df.groupby('job_category')['salary_midpoint'].describe()""")
]
create_notebook("02_eda.ipynb", cells_02)

# 3. 03_preprocessing.ipynb
cells_03 = [
    make_md_cell("# Notebook 03: Data Preprocessing Pipeline\n## Stage 3 of ML Pipeline\n\nThis notebook demonstrates:\n- Duplicate removal\n- Missing value handling\n- Job title & location normalization\n- Experience level parsing\n- Technical skill extraction via NLP regex patterns"),
    make_code_cell("""import sys
sys.path.append('..')
from src.preprocessing import clean_raw_data

df_cleaned = clean_raw_data(raw_filepath='../data/raw/jobs_raw.csv', output_filepath='../data/interim/jobs_cleaned.csv')
df_cleaned.info()""")
]
create_notebook("03_preprocessing.ipynb", cells_03)

# 4. 04_nlp_skill_extraction.ipynb
cells_04 = [
    make_md_cell("# Notebook 04: NLP Skill Extraction & Feature Engineering\n## Stage 4 of ML Pipeline\n\nExtracting skill feature vectors and TF-IDF representations from job descriptions and requirements."),
    make_code_cell("""import sys
sys.path.append('..')
from src.feature_engineering import build_features

feat_df, ts_df = build_features(
    cleaned_filepath='../data/interim/jobs_cleaned.csv',
    processed_filepath='../data/processed/classification_data.csv',
    forecast_filepath='../data/processed/f_data.csv',
    tfidf_save_path='../models/tfidf_vectorizer.pkl'
)
print(f'Featured Classification Matrix Shape: {feat_df.shape}')
print(f'Time Series Dataset Shape: {ts_df.shape}')""")
]
create_notebook("04_nlp_skill_extraction.ipynb", cells_04)

# 5. 05_classification.ipynb
cells_05 = [
    make_md_cell("# Notebook 05: Supervised ML - Job Role Classification & Salary Regression\n## Stages 5, 6, 7, 8 of ML Pipeline\n\nComparing Baseline models vs Advanced models with 5-Fold Stratified Cross-Validation and Hyperparameter Tuning."),
    make_code_cell("""import sys
sys.path.append('..')
from src.models import train_job_category_classifiers, train_salary_regressor

print('=== 1. Multi-class Job Category Classification ===')
results_df, best_model, X_tr, X_te, y_tr, y_te, cat2idx = train_job_category_classifiers(
    data_path='../data/processed/classification_data.csv',
    model_save_path='../models/best_classifier.pkl',
    cm_fig_path='../figures/confusion_matrix.png',
    dist_fig_path='../figures/job_distribution.png',
    skill_fig_path='../figures/skill_frequency.png'
)
display(results_df)"""),
    make_code_cell("""print('=== 2. Secondary ML Problem: Salary Regression ===')
sal_results_df = train_salary_regressor(
    data_path='../data/processed/classification_data.csv',
    sal_fig_path='../figures/salary_distribution.png'
)
display(sal_results_df)""")
]
create_notebook("05_classification.ipynb", cells_05)

# 6. 06_forecasting.ipynb
cells_06 = [
    make_md_cell("# Notebook 06: Time-Series Forecasting of Job Demand\n## Stage 3 Analysis: Time-Aware Validation\n\nForecasting overall monthly technology job postings in Bangladesh using Baseline Naive, ARIMA, and XGBoost Lag-based models."),
    make_code_cell("""import sys
sys.path.append('..')
from src.models import train_time_series_forecaster

ts_res_df = train_time_series_forecaster(
    forecast_path='../data/processed/f_data.csv',
    forecast_fig_path='../figures/forecast.png'
)
display(ts_res_df)""")
]
create_notebook("06_forecasting.ipynb", cells_06)

# 7. 07_error_analysis.ipynb
cells_07 = [
    make_md_cell("# Notebook 07: Error Analysis & Feature Importance\n## Stages 9 & 10 of ML Pipeline\n\nInvestigating misclassification patterns and assessing model feature importance."),
    make_code_cell("""import sys
sys.path.append('..')
import joblib
import pandas as pd
from src.models import prepare_classification_features, perform_error_analysis_and_feature_importance

saved = joblib.load('../models/best_classifier.pkl')
best_classifier = saved['model']
cat2idx = saved['cat2idx']

df = pd.read_csv('../data/processed/classification_data.csv')
X, y = prepare_classification_features(df)
y_encoded = y.map(cat2idx)

feat_imp_df, error_summary = perform_error_analysis_and_feature_importance(
    best_classifier, X, y_encoded, cat2idx,
    feat_fig_path='../figures/feature_importance.png'
)

print('--- Top 10 Feature Importances ---')
display(feat_imp_df.head(10))

print('\\n--- Top Misclassifications ---')
display(error_summary.head(10))""")
]
create_notebook("07_error_analysis.ipynb", cells_07)
