import os
import pandas as pd
from src.ranker import rank_resumes
from src.ranker import plot_rankings 
from src.report_generator import create_report

def main():
    try:
        print("Initializing Resume Ranker...")
        
        # Verify paths
        data_dir = os.path.join(os.getcwd(), 'data')
        print(f"Data directory: {data_dir}")
        
        # Verify CSV exists
        csv_path = os.path.join(data_dir, 'job_descriptions.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        
        jobs = pd.read_csv(csv_path)
        job_titles = jobs['job_title'].tolist()
        print("Available job titles:", job_titles)
        
        # Verify resumes folder
        resume_folder = os.path.join(data_dir, 'resumes')
        if not os.path.exists(resume_folder):
            raise FileNotFoundError(f"Missing folder: {resume_folder}")
            
        resume_files = os.listdir(resume_folder)
        print(f"Found {len(resume_files)} resume(s): {resume_files}")

        # Process each job title
        for job_title in job_titles:
            print(f"\n{'='*40}")
            print(f"Analyzing resumes for: {job_title}")
            
            # Rank resumes
            rankings = rank_resumes(job_title, resume_folder)
            
            if not rankings:
                print("No valid resumes could be processed")
                continue
            # Generate outputs
            plot_path = plot_rankings(rankings, job_title)
            report_path = create_report(rankings, job_title, plot_path)
            
            # Print results
            print("\nTop Candidates:")
            for i, candidate in enumerate(rankings[:3]):
                print(f"{i+1}. {candidate['file']} (Score: {candidate['score']:.1f}%)")
                print(f"   Matching Skills: {', '.join(candidate['matched_skills'][:5])}...")
                if candidate['missing_skills']:
                    print(f"   Missing Skills: {', '.join(candidate['missing_skills'][:3])}...")
            
            print(f"\nReport generated: {report_path}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Verify data/job_descriptions.csv exists")
        print("2. Check data/resumes/ contains PDF/DOCX files")
        print("3. Ensure all packages are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    main()