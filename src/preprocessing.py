import pandas as pd
import numpy as np
import re
import os

# Comprehensive list of tech skills to extract
SKILL_PATTERNS = {
    'python': r'\bpython\b',
    'javascript': r'\bjavascript\b|\bjs\b',
    'typescript': r'\btypescript\b|\bts\b',
    'react': r'\breact\b|\breactjs\b|\breact\.js\b',
    'node': r'\bnode\b|\bnodejs\b|\bnode\.js\b',
    'java': r'\bjava\b(?!script)',
    'csharp': r'\bc#\b|\bcnet\b|\b\.net\b|\bdotnet\b',
    'cpp': r'\bc\+\+\b',
    'sql': r'\bsql\b|\bpostgresql\b|\bmysql\b|\boracle\b',
    'django': r'\bdjango\b',
    'fastapi': r'\bfastapi\b',
    'express': r'\bexpress\b|\bexpressjs\b',
    'flutter': r'\bflutter\b|\bdart\b',
    'aws': r'\baws\b|\bamazon web services\b',
    'azure': r'\bazure\b',
    'docker': r'\bdocker\b',
    'kubernetes': r'\bkubernetes\b|\bk8s\b',
    'linux': r'\blinux\b|\bubuntu\b',
    'tensorflow': r'\btensorflow\b',
    'pytorch': r'\bpytorch\b',
    'scikit_learn': r'\bscikit-learn\b|\bsklearn\b',
    'pandas': r'\bpandas\b',
    'numpy': r'\bnumpy\b',
    'spark': r'\bspark\b|\bapache spark\b',
    'figma': r'\bfigma\b',
    'tableau': r'\btableau\b',
    'powerbi': r'\bpowerbi\b|\bpower bi\b',
    'git': r'\bgit\b|\bgithub\b|\bgitlab\b',
    'cybersecurity': r'\bcybersecurity\b|\bpenetration testing\b|\bethical hacking\b|\bsiem\b',
    'cisco': r'\bcisco\b|\brouting\b|\bswitching\b'
}

JOB_CATEGORY_RULES = {
    'Software Development': [
        'software engineer', 'developer', 'frontend', 'backend', 'full stack', 'flutter',
        'android', 'ios', 'python developer', 'react developer', 'node.js developer', 'java software', 'programmer'
    ],
    'Data / AI': [
        'data analyst', 'data engineer', 'machine learning', 'data scientist', 'ai engineer',
        'business intelligence', 'big data', 'nlp'
    ],
    'Cybersecurity': [
        'cybersecurity', 'security analyst', 'soc analyst', 'penetration tester', 'information security'
    ],
    'Networking / Cloud': [
        'devops', 'cloud', 'network engineer', 'system administrator', 'site reliability', 'sre', 'cisco'
    ],
    'IT Support': [
        'it support', 'desktop support', 'helpdesk', 'technician', 'it operations'
    ],
    'UI/UX': [
        'ui/ux', 'product designer', 'ux researcher', 'ui designer', 'figma'
    ]
}

def safe_to_csv(df, filepath, index=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df.to_csv(filepath, index=index)
    except PermissionError:
        import time
        time.sleep(1)
        try:
            df.to_csv(filepath, index=index)
        except PermissionError:
            temp_path = filepath + ".tmp.csv"
            df.to_csv(temp_path, index=index)
            print(f"File {filepath} is locked. Saved dataset to fallback path: {temp_path}")

def assign_job_category(title):
    title_lower = str(title).lower()
    for cat, keywords in JOB_CATEGORY_RULES.items():
        if any(kw in title_lower for kw in keywords):
            return cat
    return 'Other'

def extract_experience_info(exp_str):
    exp_str = str(exp_str).lower()
    if 'fresh' in exp_str or '0-1' in exp_str or 'intern' in exp_str:
        return 0.5, 'Entry'
    elif '1-3' in exp_str or '1 to 3' in exp_str:
        return 2.0, 'Mid'
    elif '2-4' in exp_str or '2 to 4' in exp_str:
        return 3.0, 'Mid'
    elif '3-5' in exp_str or '3 to 5' in exp_str:
        return 4.0, 'Mid'
    elif '5-8' in exp_str or '5 to 8' in exp_str:
        return 6.5, 'Senior'
    elif '5+' in exp_str or 'senior' in exp_str:
        return 6.0, 'Senior'
    elif '8+' in exp_str or 'lead' in exp_str or 'executive' in exp_str:
        return 9.0, 'Lead/Executive'
    
    match = re.search(r'(\d+)', exp_str)
    if match:
        years = float(match.group(1))
        level = 'Entry' if years < 2 else ('Mid' if years < 5 else 'Senior')
        return years, level
    return 2.0, 'Mid'

def normalize_location(loc_str):
    loc_lower = str(loc_str).lower()
    if 'dhaka' in loc_lower:
        return 'Dhaka'
    elif 'chittagong' in loc_lower or 'ctg' in loc_lower:
        return 'Chittagong'
    elif 'sylhet' in loc_lower:
        return 'Sylhet'
    elif 'rajshahi' in loc_lower:
        return 'Rajshahi'
    elif 'remote' in loc_lower:
        return 'Remote'
    else:
        return 'Other'

def extract_skills_from_text(text):
    text_lower = str(text).lower()
    found_skills = []
    for skill_name, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            found_skills.append(skill_name)
    return found_skills

def clean_raw_data(raw_filepath="data/raw/jobs_raw.csv", output_filepath="data/interim/jobs_cleaned.csv"):
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw data file not found at {raw_filepath}")
        
    df = pd.read_csv(raw_filepath)
    print(f"Loaded raw dataset with {len(df)} rows.")
    
    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=['job_title', 'company', 'location']).copy()
    print(f"Removed {initial_len - len(df)} duplicates. Remaining: {len(df)}")
    
    # Robust Datetime handling
    parsed_dates = pd.to_datetime(df['posted_date'], errors='coerce', format='mixed')
    default_date = pd.to_datetime('2024-06-01')
    df['posted_date'] = parsed_dates.fillna(default_date)
    
    df['year'] = df['posted_date'].dt.year
    df['month'] = df['posted_date'].dt.month
    df['year_month'] = df['posted_date'].dt.to_period('M').astype(str)
    
    # Job Category
    df['job_category'] = df['job_title'].apply(assign_job_category)
    
    # Location Category
    df['location_category'] = df['location'].apply(normalize_location)
    
    # Experience parsing
    exp_results = df['experience'].apply(extract_experience_info)
    df['experience_years'] = [r[0] for r in exp_results]
    df['experience_level'] = [r[1] for r in exp_results]
    
    # Numeric salary midpoint
    sal_min = pd.to_numeric(df['salary_min'], errors='coerce')
    sal_max = pd.to_numeric(df['salary_max'], errors='coerce')
    df['salary_midpoint'] = (sal_min + sal_max) / 2.0
    
    # Skill Extraction
    combined_text = df['job_title'].fillna('') + ' ' + df['job_description'].fillna('') + ' ' + df['requirements'].fillna('')
    df['extracted_skills'] = combined_text.apply(extract_skills_from_text)
    df['skills'] = df['extracted_skills'].apply(lambda s: ", ".join(s))
    df['skill_count'] = df['extracted_skills'].apply(len)
    
    safe_to_csv(df, output_filepath, index=False)
    print(f"Saved cleaned interim dataset to {output_filepath}")
    return df

if __name__ == "__main__":
    clean_raw_data()
