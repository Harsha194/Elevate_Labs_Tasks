import os
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from .resume_parser import extract_text
from .matching import enhanced_match

def rank_resumes(job_title: str, resume_folder: str) -> List[Dict[str, Any]]:
    """Rank resumes against a job description"""
    jobs = pd.read_csv(os.path.join('data', 'job_descriptions.csv'))
    
    try:
        job_desc = jobs.loc[jobs['job_title'] == job_title, 'description'].iloc[0]
    except IndexError:
        raise ValueError(f"Job title '{job_title}' not found")

    rankings = []
    for file in os.listdir(resume_folder):
        if file.lower().endswith(('.pdf', '.docx', '.txt')):
            try:
                resume_text = extract_text(os.path.join(resume_folder, file))
                result = enhanced_match(job_desc, resume_text)
                rankings.append({
                    'file': file,
                    'score': result['score'],
                    'matched_skills': result['matched_skills'],
                    'missing_skills': result['missing_skills']
                })
            except Exception as e:
                print(f"Skipping {file}: {str(e)}")
    
    return sorted(rankings, key=lambda x: x['score'], reverse=True)

def plot_rankings(rankings: List[Dict[str, Any]], job_title: str) -> str:
    """Generate ranking visualization plot"""
    df = pd.DataFrame(rankings)
    df = df.sort_values('score', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['file'], df['score'], color='skyblue')
    
    # Add score labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', va='center')
    
    plt.title(f"Resume Ranking: {job_title}")
    plt.xlabel("Match Score (%)")
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    output_path = os.path.join('reports', f"{job_title.replace(' ', '_')}_ranking.png")
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    return output_path