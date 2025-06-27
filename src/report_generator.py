from fpdf import FPDF
import os
from typing import List, Dict, Any

def create_report(rankings: List[Dict[str, Any]], job_title: str, plot_path: str) -> str:
    """Generate PDF report from rankings"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"Resume Ranking Report: {job_title}", ln=True, align='C')
    pdf.ln(10)
    
    # Ranking plot
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Resume Ranking Results:", ln=True)
    pdf.image(plot_path, x=10, y=40, w=180)
    
    # Top candidates details
    pdf.add_page()
    pdf.cell(200, 10, "Top Candidates:", ln=True)
    pdf.ln(10)
    
    for i, candidate in enumerate(rankings[:3]):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, f"{i+1}. {candidate['file']} - Score: {candidate['score']:.1f}%", ln=True)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 7, f"Matching Skills: {', '.join(candidate['matched_skills'][:10])}", ln=True)
        if candidate['missing_skills']:
            pdf.cell(200, 7, f"Missing Skills: {', '.join(candidate['missing_skills'][:5])}", ln=True)
        pdf.ln(5)
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    output_path = os.path.join('reports', f"resume_ranking_{job_title.replace(' ', '_')}.pdf")
    pdf.output(output_path)
    return output_path