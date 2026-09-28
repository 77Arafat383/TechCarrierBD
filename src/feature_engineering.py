import pandas as pd
import numpy as np
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

TOP_SKILLS = [
    'python', 'javascript', 'typescript', 'react', 'node', 'java', 'csharp',
    'sql', 'django', 'flutter', 'aws', 'docker', 'kubernetes', 'linux',
    'tensorflow', 'pytorch', 'pandas', 'spark', 'figma', 'tableau', 'powerbi', 'git'
]

def safe_to_csv(df, filepath, index=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df.to_csv(filepath, index=index)
    except PermissionError:
        print(f"Warning: {filepath} is currently open in another program. Retrying...")
        import time
        time.sleep(1)
        try:
            df.to_csv(filepath, index=index)
        except PermissionError:
            temp_path = filepath + ".tmp.csv"
            df.to_csv(temp_path, index=index)
            print(f"File {filepath} is locked. Saved dataset to fallback path: {temp_path}")

def build_features(cleaned_filepath="data/interim/jobs_cleaned.csv",
                   processed_filepath="data/processed/classification_data.csv",
                   forecast_filepath="data/processed/f_data.csv",
                   tfidf_save_path="models/tfidf_vectorizer.pkl"):
    if not os.path.exists(cleaned_filepath):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_filepath}")
        
    df = pd.read_csv(cleaned_filepath)
    print(f"Loaded cleaned dataset with {len(df)} rows.")
    
    # Parse extracted_skills if stored as string representation of list
    def parse_skills(val):
        if isinstance(val, list):
            return val
        if pd.isna(val) or not val:
            return []
        try:
            return ast.literal_eval(val)
        except Exception:
            return [s.strip() for s in str(val).split(',') if s.strip()]
            
    df['skill_list'] = df['extracted_skills'].apply(parse_skills)
    
    # 1. Binary Skill Indicator Features
    for skill in TOP_SKILLS:
        col_name = f"has_{skill}"
        df[col_name] = df['skill_list'].apply(lambda s_list: 1 if skill in s_list else 0)
        
    # 2. Text TF-IDF vectorization
    text_data = df['job_description'].fillna('') + ' ' + df['requirements'].fillna('')
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_data)
    
    os.makedirs(os.path.dirname(tfidf_save_path), exist_ok=True)
    joblib.dump(tfidf, tfidf_save_path)
    print(f"Saved TF-IDF Vectorizer to {tfidf_save_path}")
    
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()])
    
    # Concatenate features
    featured_df = pd.concat([df, tfidf_df], axis=1)
    
    safe_to_csv(featured_df, processed_filepath, index=False)
    print(f"Saved classification featured dataset to {processed_filepath}")
    
    # 3. Build Time-Series Job Demand Dataset (f_data.csv)
    # Aggregate job count per month per category
    ts_df = df.groupby(['year_month', 'job_category']).size().reset_index(name='job_count')
    ts_pivot = ts_df.pivot(index='year_month', columns='job_category', values='job_count').fillna(0)
    ts_pivot['Total_Jobs'] = ts_pivot.sum(axis=1)
    ts_pivot = ts_pivot.reset_index()
    
    # Add lag features for Total Jobs and major categories
    ts_pivot['year_month'] = pd.to_datetime(ts_pivot['year_month'])
    ts_pivot = ts_pivot.sort_values('year_month').reset_index(drop=True)
    
    ts_pivot['lag_1'] = ts_pivot['Total_Jobs'].shift(1)
    ts_pivot['lag_2'] = ts_pivot['Total_Jobs'].shift(2)
    ts_pivot['lag_3'] = ts_pivot['Total_Jobs'].shift(3)
    ts_pivot['rolling_mean_3'] = ts_pivot['Total_Jobs'].shift(1).rolling(3).mean()
    ts_pivot['rolling_mean_6'] = ts_pivot['Total_Jobs'].shift(1).rolling(6).mean()
    ts_pivot['growth_rate'] = (ts_pivot['lag_1'] - ts_pivot['lag_2']) / (ts_pivot['lag_2'] + 1e-5)
    
    safe_to_csv(ts_pivot, forecast_filepath, index=False)
    print(f"Saved forecasting dataset to {forecast_filepath}")
    
    return featured_df, ts_pivot

if __name__ == "__main__":
    build_features()
