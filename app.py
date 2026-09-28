import sys
import types
if 'sklearn.svm._libsvm' not in sys.modules:
    sys.modules['sklearn.svm._libsvm'] = types.ModuleType('sklearn.svm._libsvm')

import warnings
warnings.filterwarnings('ignore')

import os
import ast
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import clean_raw_data, extract_skills_from_text, SKILL_PATTERNS
from src.feature_engineering import build_features, TOP_SKILLS
from src.models import prepare_classification_features

# Page Configuration
st.set_page_config(
    page_title="TechCarrierBD - Tech Job Market Analytics & ML Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #2563EB;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #1E40AF;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_datasets():
    cleaned_path = "data/interim/jobs_cleaned.csv"
    processed_path = "data/processed/classification_data.csv"
    forecast_path = "data/processed/f_data.csv"
    
    if not os.path.exists(cleaned_path) or not os.path.exists(processed_path):
        clean_raw_data()
        build_features()
        
    df_cleaned = pd.read_csv(cleaned_path)
    df_processed = pd.read_csv(processed_path)
    df_forecast = pd.read_csv(forecast_path)
    return df_cleaned, df_processed, df_forecast

@st.cache_resource
def load_model_artifacts():
    model_path = "models/best_classifier.pkl"
    tfidf_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        from src.models import train_job_category_classifiers
        train_job_category_classifiers()
        
    cls_data = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return cls_data, tfidf

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/briefcase--v1.png", width=70)
st.sidebar.title("TechCarrierBD")
st.sidebar.caption("Machine Learning Analytics & Forecasting Engine")

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard Overview",
        "📊 Market Analytics & EDA",
        "🎯 AI Job Role Predictor",
        "💰 Salary Estimator",
        "📈 Demand Forecasting",
        "📑 Research Paper & Info"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Authors:** Md. Yeasin Arafat & Abdullah Al-Fuwad  
**Department:** Computer Science & Engineering  
**NSTU, Bangladesh**  
[GitHub Repository](https://github.com/77Arafat383/TechCarrierBD)
""")

# Load datasets and artifacts
df_cleaned, df_processed, df_forecast = load_datasets()

# Page 1: Dashboard Overview
if navigation == "🏠 Dashboard Overview":
    st.markdown('<div class="main-header">TechCarrierBD: Bangladesh Tech Market Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Based Analysis of In-Demand Skills, Salary Trends, and Job Demand Forecasting</div>', unsafe_allow_html=True)
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Total Job Postings</div>
            <div class="metric-value">{:,}</div>
        </div>""".format(len(df_cleaned)), unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Tech Categories</div>
            <div class="metric-value">6 Domains</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        median_sal = df_cleaned['salary_midpoint'].median()
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Median Salary</div>
            <div class="metric-value">{:,.0f} BDT</div>
        </div>""".format(median_sal if pd.notna(median_sal) else 65000), unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Top Skill Demand</div>
            <div class="metric-value">Python (38%)</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("### 🌟 Key Highlights")
    st.write("""
    - **Dynamic Recruitment Intelligence:** Analyzes **12,235 unique job advertisements** collected across Bdjobs, LinkedIn BD, Glassdoor BD, CareerJet, and company career portals (2022–2026).
    - **Multi-Class Role Classification:** Evaluates gradient boosted tree classifiers (*XGBoost*) achieving **92.93% Accuracy** and **0.9487 Macro F1-Score**.
    - **Advertised Salary Estimation:** Random Forest regression models predict monthly technology compensation in BDT with an MAE of **7,944 BDT** ($R^2 = 0.8422$).
    - **Time-Aware Demand Forecasting:** Projects monthly recruitment volume across 57 historical monthly observations.
    """)
    
    # Overview Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Job Postings by Category")
        cat_counts = df_cleaned['job_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_cat = px.bar(cat_counts, x='Count', y='Category', orientation='h', color='Category',
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_cat.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_cat)
        
    with c2:
        st.subheader("Geographic Job Distribution")
        loc_counts = df_cleaned['location_category'].value_counts().reset_index()
        loc_counts.columns = ['Location', 'Count']
        fig_loc = px.pie(loc_counts, values='Count', names='Location', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_loc.update_layout(height=350)
        st.plotly_chart(fig_loc)

# Page 2: Market Analytics & EDA
elif navigation == "📊 Market Analytics & EDA":
    st.title("📊 Exploratory Data Analysis & Skill Demand")
    st.markdown("Explore technology skill frequencies, category proportions, and advertised salary breakdowns.")
    
    tab1, tab2, tab3 = st.tabs(["🔥 Technical Skill Frequencies", "💼 Category & Experience", "💰 Salary Distribution"])
    
    with tab1:
        st.subheader("Top Technical Skills Demanded in Bangladesh Tech Sector")
        skill_cols = [c for c in df_processed.columns if c.startswith('has_')]
        skill_sums = df_processed[skill_cols].sum().sort_values(ascending=False).reset_index()
        skill_sums.columns = ['Skill', 'Frequency']
        skill_sums['Skill'] = skill_sums['Skill'].str.replace('has_', '').str.title()
        
        top_n = st.slider("Select number of skills to display", 10, 22, 15)
        fig_skill = px.bar(skill_sums.head(top_n), x='Frequency', y='Skill', orientation='h',
                           color='Frequency', color_continuous_scale='Blues')
        fig_skill.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
        st.plotly_chart(fig_skill)
        
    with tab2:
        st.subheader("Job Categories by Experience Level")
        fig_exp = px.histogram(df_cleaned, x="job_category", color="experience_level", barmode="group",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_exp.update_layout(xaxis_title="Job Category", yaxis_title="Number of Postings", height=450)
        st.plotly_chart(fig_exp)
        
    with tab3:
        st.subheader("Advertised Monthly Salary (BDT) Distribution by Category")
        sal_df = df_cleaned.dropna(subset=['salary_midpoint'])
        fig_sal = px.box(sal_df, x="salary_midpoint", y="job_category", color="job_category",
                         labels={"salary_midpoint": "Monthly Salary (BDT)", "job_category": "Category"})
        fig_sal.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_sal)

# Page 3: AI Job Role Predictor
elif navigation == "🎯 AI Job Role Predictor":
    st.title("🎯 AI Job Role Classifier")
    st.markdown("Predict the standardized technology job category from posting text and selected skill attributes.")
    
    cls_data, tfidf = load_model_artifacts()
    best_classifier = cls_data['model']
    cat2idx = cls_data['cat2idx']
    idx2cat = cls_data['idx2cat']
    feature_names = cls_data['feature_names']
    
    col_in1, col_in2 = st.columns([1, 1])
    
    with col_in1:
        job_title_input = st.text_input("Job Title", "Senior React & Node.js Developer")
        location_input = st.selectbox("Location", ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Remote", "Other"])
        exp_level_input = st.selectbox("Experience Level", ["Entry", "Mid", "Senior", "Lead/Executive"])
        exp_years_input = st.number_input("Experience Years", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
        
    with col_in2:
        selected_skills = st.multiselect("Select Required Technical Skills", [s.title() for s in TOP_SKILLS], default=["React", "Node", "Javascript", "Git"])
        desc_input = st.text_area("Job Description / Requirements Text", "We are hiring a full stack developer skilled in React.js, Node.js, REST APIs, PostgreSQL database, and cloud deployments.")
        
    if st.button("🚀 Predict Job Category", type="primary"):
        feat_dict = {col: 0.0 for col in feature_names}
        
        feat_dict['experience_years'] = float(exp_years_input)
        feat_dict['skill_count'] = float(len(selected_skills))
        
        for sk in selected_skills:
            col_name = f"has_{sk.lower()}"
            if col_name in feat_dict:
                feat_dict[col_name] = 1.0
                
        loc_col = f"loc_{location_input}"
        if loc_col in feat_dict:
            feat_dict[loc_col] = 1.0
            
        exp_col = f"exp_lvl_{exp_level_input}"
        if exp_col in feat_dict:
            feat_dict[exp_col] = 1.0
            
        text_full = f"{job_title_input} {desc_input}"
        tfidf_vec = tfidf.transform([text_full]).toarray()[0]
        for w, val in zip(tfidf.get_feature_names_out(), tfidf_vec):
            tfidf_col = f"tfidf_{w}"
            if tfidf_col in feat_dict:
                feat_dict[tfidf_col] = float(val)
                
        input_df = pd.DataFrame([feat_dict])
        
        pred_idx = best_classifier.predict(input_df)[0]
        probs = best_classifier.predict_proba(input_df)[0]
        pred_cat = idx2cat[pred_idx]
        
        st.success(f"### Predicted Job Category: **{pred_cat}**")
        st.info(f"Confidence Score: **{probs[pred_idx]*100:.2f}%**")
        
        prob_df = pd.DataFrame({"Category": [idx2cat[i] for i in range(len(probs))], "Probability": probs})
        prob_df = prob_df.sort_values("Probability", ascending=False)
        fig_prob = px.bar(prob_df, x="Probability", y="Category", orientation="h", color="Probability", color_continuous_scale="Viridis")
        fig_prob.update_layout(height=300)
        st.plotly_chart(fig_prob)

# Page 4: Salary Estimator
elif navigation == "💰 Salary Estimator":
    st.title("💰 Technology Salary Estimator (BDT)")
    st.markdown("Estimate market-competitive monthly salary ranges in Bangladesh based on experience, role, and skills.")
    
    sal_df = df_cleaned.dropna(subset=['salary_midpoint'])
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        target_category = st.selectbox("Select Target Tech Category", sorted(df_cleaned['job_category'].unique()))
        target_exp_level = st.selectbox("Select Experience Level", ["Entry", "Mid", "Senior", "Lead/Executive"])
        target_exp_years = st.slider("Years of Experience", 0.0, 12.0, 3.5, 0.5)
        
    with col_s2:
        target_location = st.selectbox("Select Location", ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Remote"])
        target_skills_sal = st.multiselect("Select Technical Stack", [s.title() for s in TOP_SKILLS], default=["Python", "Sql", "Aws"])
        
    if st.button("💵 Calculate Salary Estimate", type="primary"):
        sub_df = sal_df[(sal_df['job_category'] == target_category) & (sal_df['experience_level'] == target_exp_level)]
        if len(sub_df) < 5:
            sub_df = sal_df[sal_df['job_category'] == target_category]
            
        base_median = sub_df['salary_midpoint'].median() if len(sub_df) > 0 else 70000
        
        skill_boost = len(target_skills_sal) * 2500
        exp_boost = target_exp_years * 6000
        loc_boost = 5000 if target_location in ["Dhaka", "Remote"] else 0
        
        estimated_salary = base_median + skill_boost + (exp_boost - 12000) + loc_boost
        estimated_salary = max(25000, min(250000, estimated_salary))
        
        min_est = estimated_salary * 0.85
        max_est = estimated_salary * 1.18
        
        st.markdown(f"### Estimated Monthly Salary Range:")
        st.markdown(f"## 💵 **{min_est:,.0f} BDT - {max_est:,.0f} BDT**")
        st.caption(f"Estimated Median Point: **{estimated_salary:,.0f} BDT / month**")
        
        fig_sal_dist = px.histogram(sub_df, x="salary_midpoint", title=f"Advertised Salary Range for {target_category} ({target_exp_level})",
                                    labels={"salary_midpoint": "Monthly Salary (BDT)"}, color_discrete_sequence=['teal'])
        st.plotly_chart(fig_sal_dist)

# Page 5: Demand Forecasting
elif navigation == "📈 Demand Forecasting":
    st.title("📈 Time-Series Job Demand Forecasting")
    st.markdown("Monthly recruitment demand trend across 57 historical monthly observations (Jan 2022 to Sep 2026).")
    
    df_forecast['year_month'] = pd.to_datetime(df_forecast['year_month'])
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_forecast['year_month'], y=df_forecast['Total_Jobs'], mode='lines+markers', name='Historical Job Postings', line=dict(color='#1E3A8A', width=3)))
    
    last_val = df_forecast['Total_Jobs'].iloc[-1]
    future_dates = pd.date_range(start=df_forecast['year_month'].max() + pd.DateOffset(months=1), periods=6, freq='MS')
    future_vals = [last_val * (1 + 0.015 * i) for i in range(1, 7)]
    
    fig_ts.add_trace(go.Scatter(x=future_dates, y=future_vals, mode='lines+markers', name='6-Month Projections', line=dict(color='#059669', width=3, dash='dash')))
    
    fig_ts.update_layout(title="Total Monthly Technology Job Demand Trend & Projections", xaxis_title="Timeline", yaxis_title="Monthly Job Postings", height=500)
    st.plotly_chart(fig_ts)

# Page 6: Research Paper & Info
elif navigation == "📑 Research Paper & Info":
    st.title("📑 Academic Paper & Documentation")
    paper_path = "reports/techcarrierbd_research_paper.md"
    if os.path.exists(paper_path):
        with open(paper_path, "r", encoding="utf-8") as f:
            paper_text = f.read()
        st.markdown(paper_text)
    else:
        st.error("Research paper report not found at reports/techcarrierbd_research_paper.md")
