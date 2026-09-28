import sys
import types
if 'sklearn.svm._libsvm' not in sys.modules:
    sys.modules['sklearn.svm._libsvm'] = types.ModuleType('sklearn.svm._libsvm')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score
)
from statsmodels.tsa.arima.model import ARIMA

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def prepare_classification_features(df):
    skill_cols = [col for col in df.columns if col.startswith('has_')]
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
    
    location_dummies = pd.get_dummies(df['location_category'], prefix='loc', drop_first=True)
    exp_level_dummies = pd.get_dummies(df['experience_level'], prefix='exp_lvl', drop_first=True)
    
    num_cols = ['experience_years', 'skill_count']
    
    X = pd.concat([df[num_cols + skill_cols + tfidf_cols], location_dummies, exp_level_dummies], axis=1)
    X = X.astype(float)
    y = df['job_category']
    return X, y

def train_job_category_classifiers(data_path="data/processed/classification_data.csv",
                                    model_save_path="models/best_classifier.pkl",
                                    cm_fig_path="figures/confusion_matrix.png",
                                    dist_fig_path="figures/job_distribution.png",
                                    skill_fig_path="figures/skill_frequency.png"):
    df = pd.read_csv(data_path)
    print(f"Loaded classification dataset: {len(df)} records.")
    
    # Save Job Distribution Plot
    plt.figure(figsize=(10, 5))
    order = df['job_category'].value_counts().index
    sns.countplot(data=df, y='job_category', order=order, palette='viridis', hue='job_category', legend=False)
    plt.title("Technology Job Category Distribution in Bangladesh (TechCarrierBD)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Job Postings")
    plt.ylabel("Job Category")
    plt.tight_layout()
    os.makedirs(os.path.dirname(dist_fig_path), exist_ok=True)
    plt.savefig(dist_fig_path, dpi=300)
    plt.close()
    
    # Save Top Skills Plot
    skill_cols = [c for c in df.columns if c.startswith('has_')]
    skill_counts = df[skill_cols].sum().sort_values(ascending=False)
    skill_counts.index = [c.replace('has_', '').title() for c in skill_counts.index]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=skill_counts.values[:15], y=skill_counts.index[:15], palette='mako', hue=skill_counts.index[:15], legend=False)
    plt.title("Top 15 Most Demanded Tech Skills in Bangladesh", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Postings Mentioning Skill")
    plt.ylabel("Skill")
    plt.tight_layout()
    os.makedirs(os.path.dirname(skill_fig_path), exist_ok=True)
    plt.savefig(skill_fig_path, dpi=300)
    plt.close()

    X, y = prepare_classification_features(df)
    
    categories = sorted(y.unique())
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    y_encoded = y.map(cat2idx)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42, eval_metric='mlogloss', n_jobs=1)
    }
    
    results = []
    trained_models = {}
    
    print("\n--- Model Training & 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=['accuracy', 'f1_macro'], n_jobs=1)
        cv_acc = cv_res['test_accuracy'].mean()
        cv_f1 = cv_res['test_f1_macro'].mean()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Model": name,
            "CV Accuracy": round(cv_acc, 4),
            "CV Macro F1": round(cv_f1, 4),
            "Test Accuracy": round(acc, 4),
            "Test Precision": round(prec, 4),
            "Test Recall": round(rec, 4),
            "Test Macro F1": round(f1, 4),
            "Test Weighted F1": round(f1_weighted, 4)
        })
        trained_models[name] = model
        print(f"{name:20s} | Test Acc: {acc:.4f} | Test Macro F1: {f1:.4f}")
        
    results_df = pd.DataFrame(results)
    
    print("\n--- Hyperparameter Tuning (RandomizedSearchCV on XGBoost) ---")
    param_dist = {
        'n_estimators': [50, 80],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=1)
    rsearch = RandomizedSearchCV(xgb_base, param_distributions=param_dist, n_iter=3, cv=3,
                                 scoring='f1_macro', random_state=42, n_jobs=1)
    rsearch.fit(X_train, y_train)
    best_xgb = rsearch.best_estimator_
    
    y_pred_tuned = best_xgb.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    prec, rec, tuned_f1, _ = precision_recall_fscore_support(y_test, y_pred_tuned, average='macro', zero_division=0)
    _, _, tuned_f1_w, _ = precision_recall_fscore_support(y_test, y_pred_tuned, average='weighted', zero_division=0)
    
    results_df = pd.concat([results_df, pd.DataFrame([{
        "Model": "Tuned XGBoost",
        "CV Accuracy": round(rsearch.best_score_, 4),
        "CV Macro F1": round(rsearch.best_score_, 4),
        "Test Accuracy": round(tuned_acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall": round(rec, 4),
        "Test Macro F1": round(tuned_f1, 4),
        "Test Weighted F1": round(tuned_f1_w, 4)
    }])], ignore_index=True)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"model": best_xgb, "cat2idx": cat2idx, "idx2cat": {v: k for k, v in cat2idx.items()}, "feature_names": X.columns.tolist()}, model_save_path)
    print(f"\nSaved best model to {model_save_path}")
    
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix - Tuned XGBoost Classifier", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    os.makedirs(os.path.dirname(cm_fig_path), exist_ok=True)
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    
    return results_df, best_xgb, X_train, X_test, y_train, y_test, cat2idx

def train_salary_regressor(data_path="data/processed/classification_data.csv",
                           sal_fig_path="figures/salary_distribution.png"):
    df = pd.read_csv(data_path)
    sal_df = df.dropna(subset=['salary_midpoint']).copy()
    print(f"\nSalary Analysis & Regression: {len(sal_df)} rows with salary information.")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sal_df['salary_midpoint'] / 1000, kde=True, ax=axes[0], color='teal')
    axes[0].set_title("Advertised Salary Distribution (in '000 BDT)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Monthly Salary (1,000 BDT)")
    axes[0].set_ylabel("Count")
    
    sns.boxplot(data=sal_df, x='salary_midpoint', y='job_category', ax=axes[1], palette='Set2', hue='job_category', legend=False)
    axes[1].set_title("Salary Range by Technology Category", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Monthly Salary (BDT)")
    axes[1].set_ylabel("")
    plt.tight_layout()
    os.makedirs(os.path.dirname(sal_fig_path), exist_ok=True)
    plt.savefig(sal_fig_path, dpi=300)
    plt.close()
    
    X, _ = prepare_classification_features(sal_df)
    cat_dummies = pd.get_dummies(sal_df['job_category'], prefix='cat', drop_first=True)
    X_reg = pd.concat([X, cat_dummies], axis=1).astype(float)
    y_reg = sal_df['salary_midpoint']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)
    
    reg_models = {
        "Baseline Median Regressor": DummyRegressor(strategy="median"),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
        "XGBoost Regressor": XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=1)
    }
    
    reg_results = []
    for name, reg in reg_models.items():
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        rmse = root_mean_squared_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        reg_results.append({
            "Model": name,
            "MAE (BDT)": round(mae, 2),
            "RMSE (BDT)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        print(f"{name:30s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:.4f}")
        
    return pd.DataFrame(reg_results)

def train_time_series_forecaster(forecast_path="data/processed/f_data.csv",
                                forecast_fig_path="figures/forecast.png"):
    df = pd.read_csv(forecast_path)
    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('year_month').reset_index(drop=True)
    
    print(f"\nTime-Series Forecasting: {len(df)} monthly observations ({df['year_month'].min().strftime('%Y-%m')} to {df['year_month'].max().strftime('%Y-%m')})")
    
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train period: {train_df['year_month'].min().strftime('%Y-%m')} to {train_df['year_month'].max().strftime('%Y-%m')} ({len(train_df)} months)")
    print(f"Test period:  {test_df['year_month'].min().strftime('%Y-%m')} to {test_df['year_month'].max().strftime('%Y-%m')} ({len(test_df)} months)")
    
    naive_preds = test_df['lag_1'].values
    
    train_series = train_df['Total_Jobs'].values
    test_series = test_df['Total_Jobs'].values
    
    try:
        arima_model = ARIMA(train_series, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_preds = arima_fit.forecast(steps=len(test_series))
    except Exception as e:
        print(f"ARIMA fit error: {e}, falling back to order (1, 0, 0)")
        arima_model = ARIMA(train_series, order=(1, 0, 0))
        arima_fit = arima_model.fit()
        arima_preds = arima_fit.forecast(steps=len(test_series))
        
    lag_features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6', 'growth_rate']
    train_clean = train_df.dropna(subset=lag_features).copy()
    test_clean = test_df.dropna(subset=lag_features).copy()
    
    xgb_ts = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)
    xgb_ts.fit(train_clean[lag_features], train_clean['Total_Jobs'])
    xgb_preds = xgb_ts.predict(test_clean[lag_features])
    
    def calc_ts_metrics(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {"Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}
        
    test_actuals = test_clean['Total_Jobs'].values
    naive_eval = test_clean['lag_1'].values
    arima_eval = arima_preds[:len(test_clean)]
    
    ts_results = [
        calc_ts_metrics(test_actuals, naive_eval, "Baseline Naive Forecast"),
        calc_ts_metrics(test_actuals, arima_eval, "ARIMA(1,1,1) Forecast"),
        calc_ts_metrics(test_actuals, xgb_preds, "XGBoost Lag-Based Forecast")
    ]
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['year_month'], train_df['Total_Jobs'], label='Historical Training Data', color='navy', linewidth=2)
    plt.plot(test_df['year_month'], test_df['Total_Jobs'], label='Actual Test Market Demand', color='black', linewidth=2.5, linestyle='--')
    plt.plot(test_clean['year_month'], arima_eval, label='ARIMA Forecast', color='orange', linewidth=2)
    plt.plot(test_clean['year_month'], xgb_preds, label='XGBoost Lag Model Forecast', color='green', linewidth=2)
    
    plt.title("Bangladesh Technology Job Demand Monthly Forecasting (2022-2026)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline (Month-Year)")
    plt.ylabel("Total Monthly Technology Job Postings")
    plt.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(forecast_fig_path), exist_ok=True)
    plt.savefig(forecast_fig_path, dpi=300)
    plt.close()
    
    return pd.DataFrame(ts_results)

def perform_error_analysis_and_feature_importance(best_classifier, X_test, y_test, cat2idx,
                                                  feat_fig_path="figures/feature_importance.png"):
    idx2cat = {v: k for k, v in cat2idx.items()}
    y_preds = best_classifier.predict(X_test)
    
    importances = best_classifier.feature_importances_
    feat_names = X_test.columns.tolist()
    
    clean_names = []
    for fn in feat_names:
        if fn.startswith('has_'):
            clean_names.append(f"Skill: {fn.replace('has_', '').title()}")
        elif fn.startswith('tfidf_'):
            clean_names.append(f"TFIDF: {fn.replace('tfidf_', '')}")
        elif fn.startswith('loc_'):
            clean_names.append(f"Location: {fn.replace('loc_', '')}")
        elif fn.startswith('exp_lvl_'):
            clean_names.append(f"Exp Level: {fn.replace('exp_lvl_', '')}")
        else:
            clean_names.append(fn.replace('_', ' ').title())
            
    feat_imp_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='crest', hue='Feature', legend=False)
    plt.title("Top 15 Most Important Features in Job Category Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("Feature Importance Score (Gini / Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    os.makedirs(os.path.dirname(feat_fig_path), exist_ok=True)
    plt.savefig(feat_fig_path, dpi=300)
    plt.close()
    
    test_indices = y_test.index
    df_processed = pd.read_csv("data/processed/classification_data.csv").loc[test_indices].copy()
    df_processed['Actual_Category'] = y_test.map(idx2cat)
    df_processed['Predicted_Category'] = [idx2cat[p] for p in y_preds]
    
    errors_df = df_processed[df_processed['Actual_Category'] != df_processed['Predicted_Category']]
    error_summary = errors_df.groupby(['Actual_Category', 'Predicted_Category']).size().reset_index(name='Error_Count')
    error_summary = error_summary.sort_values('Error_Count', ascending=False)
    
    return feat_imp_df, error_summary

if __name__ == "__main__":
    cls_res, best_cls, X_tr, X_te, y_tr, y_te, cat2idx = train_job_category_classifiers()
    sal_res = train_salary_regressor()
    ts_res = train_time_series_forecaster()
    feat_df, err_df = perform_error_analysis_and_feature_importance(best_cls, X_te, y_te, cat2idx)
