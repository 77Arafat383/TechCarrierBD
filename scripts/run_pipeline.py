import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import joblib

from src.preprocessing import clean_raw_data
from src.feature_engineering import build_features
from src.models import (
    train_job_category_classifiers,
    train_salary_regressor,
    train_time_series_forecaster,
    prepare_classification_features,
    perform_error_analysis_and_feature_importance
)

def run_all():
    print("==================================================")
    print(" TECHCARRIERBD: ML PIPELINE EXECUTION ")
    print("==================================================")
    
    # Step 1: Preprocessing & Cleaning
    print("\n--- STEP 1: Cleaning & Normalization ---")
    df_cleaned = clean_raw_data()
    
    # Step 2: NLP & Feature Engineering
    print("\n--- STEP 2: Feature Engineering & Time-Series Aggregation ---")
    feat_df, ts_df = build_features()
    
    # Step 3: Multi-class Job Category Classification
    print("\n--- STEP 3: Multi-class Classification Experiments ---")
    cls_results, best_model, X_tr, X_te, y_tr, y_te, cat2idx = train_job_category_classifiers()
    print("\nClassification Results Summary:")
    print(cls_results.to_string(index=False))
    
    # Step 4: Salary Regression
    print("\n--- STEP 4: Salary Regression Experiments ---")
    sal_results = train_salary_regressor()
    print("\nSalary Regression Summary:")
    print(sal_results.to_string(index=False))
    
    # Step 5: Time-Series Forecasting
    print("\n--- STEP 5: Time-Series Demand Forecasting ---")
    ts_results = train_time_series_forecaster()
    print("\nTime-Series Forecasting Summary:")
    print(ts_results.to_string(index=False))
    
    # Step 6: Error Analysis & Feature Importance
    print("\n--- STEP 6: Error Analysis & Feature Importance ---")
    feat_imp, err_summary = perform_error_analysis_and_feature_importance(best_model, X_te, y_te, cat2idx)
    print("\nTop 10 Important Features:")
    print(feat_imp.head(10).to_string(index=False))
    print("\nTop Misclassifications:")
    print(err_summary.head(10).to_string(index=False))
    
    print("\n==================================================")
    print(" PIPELINE SUCCESSFULLY COMPLETED ")
    print("==================================================")

if __name__ == "__main__":
    run_all()
