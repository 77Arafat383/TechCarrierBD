import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("scripts", exist_ok=True)

# Companies in BD Tech Scene
companies = [
    "Brain Station 23", "Enosis Solutions", "Kaz Software", "BJIT Group", "Therap Java",
    "TigerIT Bangladesh", "Selise Digital Platforms", "bKash Limited", "Pathao", "Chaldal",
    "Portonics", "Kona Software Lab", "Nascenia", "Reve Systems", "Dynamic Solution Innovators",
    "Vivasoft Ltd", "Augmedix Bangladesh", "DataSoft Systems", "Cefalo Bangladesh", "NewsCred/Welcome",
    "Appsoffice", "RedGrocer Tech", "Bondstein Tech", "ShopUp", "Field Information Solutions",
    "IPDC Finance Tech", "Standard Chartered BD Tech", "Brac IT Services", "Grameenphone IT", "Robi Axiata Digital"
]

# Locations
locations = ["Dhaka", "Dhaka", "Dhaka", "Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Remote (Bangladesh)"]

# Job Templates per Category
job_categories = {
    "Software Development": {
        "titles": [
            "Software Engineer", "Senior Software Engineer", "Full Stack Developer", "Backend Software Engineer",
            "Frontend Developer", "Python Developer", "React Developer", "Node.js Developer", "Java Software Engineer",
            "DotNet Software Engineer", "Mobile App Developer (Flutter)", "iOS Developer", "Android Developer"
        ],
        "skills_pool": ["Python", "JavaScript", "TypeScript", "React", "Node.js", "Java", "C#", ".NET Core", "Django", "FastAPI", "Express.js", "Flutter", "REST API", "GraphQL", "Git", "Docker", "PostgreSQL", "MongoDB"],
        "salaries": (40000, 180000),
        "weight": 0.42
    },
    "Data / AI": {
        "titles": [
            "Data Analyst", "Senior Data Engineer", "Machine Learning Engineer", "Data Scientist",
            "AI Engineer", "Business Intelligence Analyst", "Big Data Engineer", "NLP Specialist"
        ],
        "skills_pool": ["Python", "SQL", "R", "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "PyTorch", "Tableau", "PowerBI", "PostgreSQL", "Apache Spark", "Airflow", "NLP", "Computer Vision", "Docker", "AWS"],
        "salaries": (45000, 200000),
        "weight": 0.18
    },
    "Cybersecurity": {
        "titles": [
            "Cybersecurity Specialist", "Information Security Analyst", "SOC Analyst", "Penetration Tester",
            "Security Engineer", "Network & Security Administrator"
        ],
        "skills_pool": ["Cybersecurity", "Network Security", "Penetration Testing", "Ethical Hacking", "Wireshark", "SIEM", "Firewall", "Linux", "Python", "Incident Response", "ISO 27001", "OWASP"],
        "salaries": (50000, 190000),
        "weight": 0.10
    },
    "Networking / Cloud": {
        "titles": [
            "DevOps Engineer", "Cloud Infrastructure Engineer", "Network Engineer", "System Administrator",
            "Site Reliability Engineer (SRE)", "AWS Cloud Architect"
        ],
        "skills_pool": ["AWS", "Azure", "Docker", "Kubernetes", "Linux", "Terraform", "CI/CD", "Jenkins", "Ansible", "Bash", "Python", "Cisco", "Routing & Switching", "Network Administration"],
        "salaries": (45000, 195000),
        "weight": 0.15
    },
    "IT Support": {
        "titles": [
            "IT Support Engineer", "Desktop Support Specialist", "System Support Technician", "Helpdesk Administrator",
            "IT Operations Associate"
        ],
        "skills_pool": ["Hardware Support", "Windows Server", "Active Directory", "Network Troubleshooting", "Linux", "Helpdesk", "ITIL", "Office 365", "System Maintenance"],
        "salaries": (25000, 65000),
        "weight": 0.08
    },
    "UI/UX": {
        "titles": [
            "UI/UX Designer", "Senior Product Designer", "UX Researcher", "UI Designer & Developer", "Figma Specialist"
        ],
        "skills_pool": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research", "Usability Testing", "HTML/CSS", "Design Systems", "User Centered Design"],
        "salaries": (35000, 140000),
        "weight": 0.07
    }
}

experiences = [
    ("0-1 years (Fresh Graduate)", 0.5, "Entry"),
    ("1-3 years", 2.0, "Mid"),
    ("2-4 years", 3.0, "Mid"),
    ("3-5 years", 4.0, "Mid"),
    ("5+ years", 6.0, "Senior"),
    ("5-8 years", 6.5, "Senior"),
    ("8+ years", 9.0, "Lead/Executive")
]

educations = [
    "BSc in Computer Science & Engineering (CSE)",
    "BSc in Software Engineering",
    "BSc in EEE / IT / Related Field",
    "Bachelor's Degree in Computer Science or Equivalent",
    "MSc in Data Science / Computer Science",
    "Diploma in Engineering (Computer / IT)"
]

employment_types = ["Full-time", "Full-time", "Full-time", "Contractual", "Part-time", "Internship"]
sources = ["Bdjobs", "Bdjobs", "LinkedIn BD", "LinkedIn BD", "Glassdoor BD", "Company Career Portal"]

# Generate timestamps from Jan 2022 to Sep 2026 (57 months)
start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 9, 25)

def get_random_date(start, end):
    delta = (end - start).days
    u = random.random() ** 0.8
    random_days = int(u * delta)
    return start + timedelta(days=random_days)

records = []
N_POSTINGS = 12500

categories_list = list(job_categories.keys())
category_weights = [job_categories[c]["weight"] for c in categories_list]

for i in range(1, N_POSTINGS + 1):
    job_id = f"BDJ-{i:06d}"
    cat_name = random.choices(categories_list, weights=category_weights)[0]
    cat_info = job_categories[cat_name]
    
    title = random.choice(cat_info["titles"])
    company = random.choice(companies)
    location = random.choice(locations)
    posted_date = get_random_date(start_date, end_date)
    
    exp_text, exp_years, exp_level = random.choice(experiences)
    
    if exp_level == "Senior" and not ("Senior" in title or "Lead" in title or "Architect" in title):
        if random.random() > 0.4:
            title = f"Senior {title}"
    elif exp_level == "Lead/Executive" and not ("Lead" in title or "Architect" in title or "Manager" in title):
        title = f"Lead {title}"
    elif exp_level == "Entry" and ("Senior" in title or "Lead" in title):
        title = title.replace("Senior ", "").replace("Lead ", "")
        
    education = random.choice(educations)
    emp_type = random.choice(employment_types)
    source = random.choice(sources)
    url_id = f"https://www.bdjobs.com/jobdetails.asp?id={1000000 + i}" if source == "Bdjobs" else f"https://www.linkedin.com/jobs/view/{2000000000 + i}"
    
    n_skills = random.randint(3, 8)
    chosen_skills = random.sample(cat_info["skills_pool"], min(n_skills, len(cat_info["skills_pool"])))
    
    base_min, base_max = cat_info["salaries"]
    exp_multiplier = 0.7 + (exp_years / 5.0) * 0.8
    
    sal_min_val = int(base_min * exp_multiplier * random.uniform(0.85, 1.15) / 1000) * 1000
    sal_max_val = int((sal_min_val + random.uniform(15000, 45000)) / 1000) * 1000
    
    if random.random() < 0.35:
        salary_min = np.nan
        salary_max = np.nan
    else:
        salary_min = float(sal_min_val)
        salary_max = float(sal_max_val)
        
    skills_str = ", ".join(chosen_skills)
    job_desc = (
        f"We are looking for a skilled {title} to join our tech team at {company} in {location}. "
        f"The candidate will be responsible for building high quality, scalable software solutions, "
        f"collaborating with cross-functional teams, and maintaining robust system architecture. "
        f"Key technical requirements include proficiency in {skills_str}. "
        f"Minimum experience required: {exp_text}. Educational qualification: {education}."
    )
    
    requirements_text = (
        f"Requirements:\n"
        f"- Minimum {exp_years} years of relevant experience in {cat_name}.\n"
        f"- Strong proficiency in {skills_str}.\n"
        f"- Degree in {education}.\n"
        f"- Excellent problem-solving skills, communication skills, and agile teamwork.\n"
        f"- Hands-on expertise with industry best practices and clean code standards."
    )
    
    records.append({
        "job_id": job_id,
        "job_title": title,
        "company": company,
        "location": location,
        "posted_date": posted_date.strftime("%Y-%m-%d"),
        "experience": exp_text,
        "education": education,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "employment_type": emp_type,
        "job_description": job_desc,
        "requirements": requirements_text,
        "source": source,
        "url_id": url_id
    })

df = pd.DataFrame(records)
df.to_csv("data/raw/jobs_raw.csv", index=False)
print(f"Generated {len(df)} postings saved to data/raw/jobs_raw.csv")
