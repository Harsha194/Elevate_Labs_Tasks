import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

SKILL_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning"
}

def enhanced_match(job_desc, resume_text):
    # Extract and normalize skills
    job_skills = extract_skills(job_desc)
    resume_skills = extract_skills(resume_text)
    
    # Expand synonyms
    job_skills = expand_synonyms(job_skills)
    resume_skills = expand_synonyms(resume_skills)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    job_vec = vectorizer.fit_transform([" ".join(job_skills)])
    resume_vec = vectorizer.transform([" ".join(resume_skills)])
    
    # Calculate similarity
    similarity = cosine_similarity(job_vec, resume_vec)[0][0]
    
    return {
        'score': round(similarity * 100, 2),
        'matched_skills': sorted(job_skills & resume_skills),
        'missing_skills': sorted(job_skills - resume_skills)
    }

def expand_synonyms(skills):
    expanded = set()
    for skill in skills:
        expanded.add(skill)
        if skill in SKILL_SYNONYMS:
            expanded.add(SKILL_SYNONYMS[skill])
    return expanded

def extract_skills(text):
    doc = nlp(text.lower())
    skills = set()
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3:
            skills.add(chunk.lemma_.strip())
    
    # Add individual nouns
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
            skills.add(token.lemma_)
    
    return skills