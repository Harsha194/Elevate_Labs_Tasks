import PyPDF2
import docx
import os

def extract_text(filepath):
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in reader.pages)
    
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        return " ".join(para.text for para in doc.paragraphs)
    
    else:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()