#!/usr/bin/env python

import pymupdf
import zipfile
import os
import re
import pandas as pd
import spacy
import language_tool_python
import datetime
import json
import atexit
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import tempfile
from PyPDF2 import PdfReader
from io import BytesIO


# Global singleton for LanguageTool
_language_tool_instance = None


def get_language_tool():
    """Lazily initialize and return a singleton LanguageTool instance."""
    global _language_tool_instance
    if _language_tool_instance is None:
        _language_tool_instance = language_tool_python.LanguageTool('en-US')
    return _language_tool_instance


def cleanup_language_tool():
    """Clean up LanguageTool resources."""
    global _language_tool_instance
    if _language_tool_instance is not None:
        _language_tool_instance.close()
        _language_tool_instance = None


# Register cleanup function to run at exit
atexit.register(cleanup_language_tool)


# Default scoring weights and thresholds
DEFAULT_WEIGHTS = {
    # Technical Score weights
    "technical_skills_weight": 0.4,
    "technical_experience_weight": 0.3,
    "technical_education_weight": 0.3,

    # Managerial Score weights
    "managerial_soft_skills_weight": 0.3,
    "managerial_achievement_weight": 0.4,
    "managerial_leadership_weight": 0.3,

    # Overall Score weights
    "overall_technical_weight": 0.4,
    "overall_managerial_weight": 0.3,
    "overall_quality_weight": 0.3,

    # Final Score weights
    "final_overall_weight": 0.7,
    "final_tfidf_weight": 0.3,

    # Thresholds and caps
    "skill_cap": 10,
    "experience_cap_technical": 10,
    "experience_cap_leadership": 15,

    # Brevity thresholds
    "brevity_min_words": 200,
    "brevity_max_words": 1000,
    "brevity_optimal_words": 600,

    # Education level scores
    "education_phd": 1.0,
    "education_postgraduate": 0.8,
    "education_bachelor": 0.6,
    "education_associate": 0.4,
    "education_other": 0.2,
}

# Final output columns for ranked results
FINAL_COLUMNS = [
    "ID",
    "Final_Score",
    "Overall(featured)_Score",
    "TF-IDF_Score",
    "Education_Level",
    "Technical_Score",
    "Managerial_Score",
    "Spell_Check_Ratio",
    "Section_Score",
    "Brevity_Score",
    "Years_of_Experience",
    "Skill_Count",
    "Extracted_Skills",
]

# Human-readable column names for display
DISPLAY_COLUMN_NAMES = {
    "ID": "ID",
    "Final_Score": "Final Score",
    "Overall(featured)_Score": "Overall Score",
    "TF-IDF_Score": "Job Match Score",
    "Education_Level": "Education Level",
    "Technical_Score": "Technical Score",
    "Managerial_Score": "Managerial Score",
    "Spell_Check_Ratio": "Spell Check Ratio",
    "Section_Score": "Section Score",
    "Brevity_Score": "Brevity Score",
    "Years_of_Experience": "Years of Experience",
    "Skill_Count": "Skill Count",
    "Extracted_Skills": "Extracted Skills",
}


def _load_default_skills() -> List[str]:
    """Return the default skills list."""
    return [
        "communication",
        "teamwork",
        "leadership",
        "problem-solving",
        "time management",
        "analytical skills",
        "creativity",
        "adaptability",
        "programming",
        "data analysis",
        "project management",
        "software development",
        "database management",
        "web development",
        "Python",
        "Java",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "SQL",
        "C++",
        "JavaScript",
        "Data Science",
        "TensorFlow",
        "PyTorch",
        "Linux",
        "Docker",
        "Kubernetes",
        "Git",
        "REST API",
        "Flask",
        "Django",
        "BERT",
        "Transformers",
        "Siamese",
        "Neural Networks",
    ]


def load_job_skills(file_path: str) -> List[str]:
    """Load general job skills from a JSON file or use a default list."""
    default_skills = _load_default_skills()

    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                skills_data = json.load(file)
            if isinstance(skills_data, list):
                return skills_data
            elif isinstance(skills_data, dict) and "skills" in skills_data:
                return skills_data["skills"]
            else:
                return default_skills
        else:
            return default_skills
    except json.JSONDecodeError:
        return default_skills


# Load spaCy NLP model at module level
nlp = spacy.load("en_core_web_sm")

# Load general skills at module level
general_skills = load_job_skills("job_skills.json")


def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    with pymupdf.open(pdf_path) as pdf:
        text = ""
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
        return text


def extract_from_zip(zip_file) -> Tuple[pd.DataFrame, Dict[str, bytes]]:
    """
    Extract text and PDF bytes from a zip archive containing PDF resumes.

    Args:
        zip_file: Path to zip file, bytes object, or file-like object

    Returns:
        Tuple of (DataFrame with ID and Text columns, dict mapping candidate_id to PDF bytes)
    """
    # Determine if zip_file is a path, bytes object, or file-like object
    temp_zip_path = None
    if isinstance(zip_file, str):
        zip_file_path = zip_file
    elif isinstance(zip_file, bytes):
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file)
        temp_zip.close()
        zip_file_path = temp_zip.name
        temp_zip_path = zip_file_path
    elif hasattr(zip_file, "read"):
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file.read())
        temp_zip.close()
        zip_file_path = temp_zip.name
        temp_zip_path = zip_file_path
    else:
        raise ValueError(
            "zip_file must be a file path, bytes object, or file-like object."
        )

    data = []
    pdf_cache = {}

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            pdf_files = [
                f
                for f in zip_ref.namelist()
                if f.lower().endswith(".pdf") and not f.startswith("__MACOSX/")
            ]
            if not pdf_files:
                raise ValueError("No PDF files found in the zip archive.")

            for pdf_file_name in pdf_files:
                with zip_ref.open(pdf_file_name) as pdf_file:
                    pdf_bytes = pdf_file.read()
                    pdf_stream = BytesIO(pdf_bytes)
                    pdf_reader = PdfReader(pdf_stream)

                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text

                    # Extract candidate ID from filename
                    base_name = os.path.splitext(os.path.basename(pdf_file_name))[0]
                    match = re.match(r"candidate_(\d+)", base_name, re.IGNORECASE)
                    if match:
                        candidate_id = int(match.group(1))
                    else:
                        # Try to parse as integer, otherwise use base_name
                        try:
                            candidate_id = int(base_name)
                        except ValueError:
                            candidate_id = base_name

                    # Store PDF bytes in cache using normalized string key
                    cache_key = str(candidate_id)
                    pdf_cache[cache_key] = pdf_bytes

                    # Clean text and add to data
                    clean_text = " ".join(text.split())
                    data.append({
                        "ID": candidate_id,
                        "Text": clean_text
                    })
    finally:
        # Clean up temporary zip file if one was created
        if temp_zip_path is not None:
            os.remove(temp_zip_path)

    return pd.DataFrame(data), pdf_cache


def preprocess_text(text):
    """Preprocess text using spaCy for tokenization and lemmatization."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens


def extract_years_of_experience(text):
    """Extract years of experience from the resume text."""
    years = re.findall(r"\b(19[7-9]\d|20[0-2]\d)\b", text)
    if len(years) >= 2:
        earliest_year = min(int(year) for year in years)
        latest_year = max(int(year) for year in years)
        current_year = datetime.datetime.now().year
        if latest_year > current_year:
            latest_year = current_year
        return latest_year - earliest_year
    return 0


def detect_education_level(text):
    """Detect the highest education level mentioned in the resume."""
    education_patterns = {
        "PhD": r"\bPh\.?D\.?\b|\bDoctor(ate)?\b",
        "Postgraduate": r"\bM\.?S\.?\b|\bM\.?A\.?\b|\bM\.?Tech\b|\bM\.?Sc\b|\bMaster(s)?\b|\bPost\s?Graduation\b|\bPostgraduate\b",
        "Bachelor": r"\bB\.?S\.?\b|\bB\.?A\.?\b|\bB\.?Tech\b|\bB\.?Sc\b|\bBachelor(s)?\b",
    }

    for level, pattern in education_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return "Other"


def calculate_spell_check_ratio(text):
    """Calculate the ratio of potential spelling errors to total words."""
    total_words = len(text.split())
    if total_words == 0:
        return 0.0

    tool = get_language_tool()
    matches = tool.check(text)
    ratio = 1 - (len(matches) / total_words)
    return max(0.0, min(1.0, ratio))


def identify_resume_sections(text):
    """Identify and score the presence of important resume sections."""
    important_sections = [
        "education",
        "experience",
        "skills",
        "projects",
        "achievements",
    ]
    optional_sections = ["summary", "objective", "interests", "activities"]
    unnecessary_sections = ["references"]

    section_score = 0
    for section in important_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score += 1

    for section in optional_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score += 0.5

    for section in unnecessary_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score -= 0.5

    return min(section_score / len(important_sections), 1)


def quantify_brevity(text, weights: Dict[str, Any] = None):
    """Quantify the brevity of the resume."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    min_words = weights.get("brevity_min_words", 200)
    max_words = weights.get("brevity_max_words", 1000)
    optimal_words = weights.get("brevity_optimal_words", 600)

    word_count = len(text.split())
    if word_count < min_words:
        return 0.5
    elif word_count > max_words:
        return 0.5
    else:
        range_size = (max_words - min_words) / 2
        return 1 - (abs(optimal_words - word_count) / range_size)


def calculate_word_sentence_counts(text):
    """Calculate word count and sentence count."""
    sentences = re.split(r"[.!?]+", text)
    word_count = len(text.split())
    sentence_count = len([s for s in sentences if s.strip()])
    return word_count, sentence_count


def calculate_skill_match_score(resume_skills, job_skills):
    """Calculate the skill match score."""
    if not job_skills:
        return 0
    matched_skills = set(resume_skills) & set(job_skills)
    return len(matched_skills) / len(job_skills)


def analyze_sentiment(text):
    """Analyze the sentiment of achievement statements in the resume."""
    blob = TextBlob(text)
    return blob.sentiment.polarity


def quantify_achievement_impact(text):
    """Quantify the impact of achievements."""
    impact_score = 0
    achievements = re.findall(
        r"\b(increased|decreased|improved|reduced|saved|generated|expanded).*?(\d+(?:\.\d+)?%?)",
        text,
        re.IGNORECASE,
    )
    for _, value in achievements:
        if "%" in value:
            impact_score += float(value.strip("%")) / 100
        else:
            impact_score += float(value) / 1000
    return min(impact_score, 1)


def calculate_technical_score(row, weights: Dict[str, Any] = None):
    """Calculate the technical CV score."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    skill_cap = weights.get("skill_cap", 10)
    experience_cap = weights.get("experience_cap_technical", 10)

    skill_count = min(len(row["Extracted_Skills"]), skill_cap)
    experience_score = min(row["Years_of_Experience"] / experience_cap, 1)

    education_scores = {
        "PhD": weights.get("education_phd", 1.0),
        "Postgraduate": weights.get("education_postgraduate", 0.8),
        "Master": weights.get("education_postgraduate", 0.8),
        "Bachelor": weights.get("education_bachelor", 0.6),
        "Associate": weights.get("education_associate", 0.4),
        "Other": weights.get("education_other", 0.2),
    }
    education_score = education_scores.get(row["Education_Level"], weights.get("education_other", 0.2))

    skills_weight = weights.get("technical_skills_weight", 0.4)
    experience_weight = weights.get("technical_experience_weight", 0.3)
    education_weight = weights.get("technical_education_weight", 0.3)

    return (skill_count / skill_cap * skills_weight +
            experience_score * experience_weight +
            education_score * education_weight)


def calculate_managerial_score(row, weights: Dict[str, Any] = None):
    """Calculate the managerial CV score."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    experience_cap = weights.get("experience_cap_leadership", 15)

    soft_skills_score = analyze_sentiment(row["Text"])
    achievement_impact = quantify_achievement_impact(row["Text"])
    leadership_score = min(row["Years_of_Experience"] / experience_cap, 1)

    soft_skills_weight = weights.get("managerial_soft_skills_weight", 0.3)
    achievement_weight = weights.get("managerial_achievement_weight", 0.4)
    leadership_weight = weights.get("managerial_leadership_weight", 0.3)

    return (soft_skills_score * soft_skills_weight +
            achievement_impact * achievement_weight +
            leadership_score * leadership_weight)


def calculate_overall_score(row, weights: Dict[str, Any] = None):
    """Calculate the overall CV score."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    technical_score = row["Technical_Score"]
    managerial_score = row["Managerial_Score"]
    resume_quality_score = (
        row["Spell_Check_Ratio"] + row["Section_Score"] + row["Brevity_Score"]
    ) / 3

    technical_weight = weights.get("overall_technical_weight", 0.4)
    managerial_weight = weights.get("overall_managerial_weight", 0.3)
    quality_weight = weights.get("overall_quality_weight", 0.3)

    return (technical_score * technical_weight +
            managerial_score * managerial_weight +
            resume_quality_score * quality_weight)


def job_description_matching(resume_text: str, job_description: str) -> float:
    """Calculate similarity between resume and job description."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def match_resume_to_job_description(resume_text, job_description):
    """Match a resume to a specific job description and return adjusted scores."""
    match_score = job_description_matching(resume_text, job_description)
    return {"Job_Match_Score": match_score}


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using NLP techniques."""
    if not text:
        return []

    doc = nlp(text)
    keyword_skills = set()
    ner_skills = set()

    for skill in general_skills:
        if skill.lower() in text.lower():
            keyword_skills.add(skill)

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART"}:
            ner_skills.add(ent.text)

    sorted_keyword_skills = sorted(keyword_skills)
    sorted_ner_skills = sorted(ner_skills)

    return sorted_keyword_skills + sorted_ner_skills


def process_resume(row, weights: Dict[str, Any] = None):
    """Process a single resume and return a dictionary of features."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    text = row["Text"]

    return {
        "Years_of_Experience": extract_years_of_experience(text),
        "Education_Level": detect_education_level(text),
        "Spell_Check_Ratio": calculate_spell_check_ratio(text),
        "Section_Score": identify_resume_sections(text),
        "Brevity_Score": quantify_brevity(text, weights),
        "Extracted_Skills": extract_skills(text),
    }


def load_and_process_resumes(
    resume_directory: str,
    weights: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, bytes]]:
    """
    Load resumes from zip and process them to extract features.

    Args:
        resume_directory: Path to zip file containing PDF resumes
        weights: Optional dictionary of scoring weights and thresholds

    Returns:
        Tuple of (DataFrame with resume data and features, PDF cache dict)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    df, pdf_cache = extract_from_zip(resume_directory)

    df["processed"] = df.apply(lambda row: process_resume(row, weights), axis=1)
    df = pd.concat([df, pd.DataFrame(df["processed"].tolist())], axis=1)
    df.drop("processed", axis=1, inplace=True)

    return df, pdf_cache


def calculate_all_scores(
    df: pd.DataFrame,
    job_description: str = None,
    weights: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Calculate all scores for the resumes.

    Args:
        df: DataFrame with resume data and extracted features
        job_description: Optional job description text for matching
        weights: Optional dictionary of scoring weights and thresholds

    Returns:
        DataFrame with all scores calculated
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    df["Skill_Count"] = df["Extracted_Skills"].apply(len)
    df["Technical_Score"] = df.apply(lambda row: calculate_technical_score(row, weights), axis=1)
    df["Managerial_Score"] = df.apply(lambda row: calculate_managerial_score(row, weights), axis=1)
    df["Overall(featured)_Score"] = df.apply(lambda row: calculate_overall_score(row, weights), axis=1)

    df["TF-IDF_Score"] = df.apply(
        lambda row: (
            match_resume_to_job_description(row["Text"], job_description).get("Job_Match_Score", 1.0)
            if job_description
            else 1.0
        ),
        axis=1,
    )

    final_overall_weight = weights.get("final_overall_weight", 0.7)
    final_tfidf_weight = weights.get("final_tfidf_weight", 0.3)
    df["Final_Score"] = df["Overall(featured)_Score"] * final_overall_weight + df["TF-IDF_Score"] * final_tfidf_weight

    return df


def rank_and_export(df: pd.DataFrame, output_path: str = "final_ranked_resumes.csv") -> pd.DataFrame:
    """
    Rank resumes by final score and export to CSV.

    Args:
        df: DataFrame with all scores calculated
        output_path: Path to save the CSV file

    Returns:
        DataFrame with ranked resumes (only final columns)
    """
    ranked_df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    ranked_df[FINAL_COLUMNS].to_csv(output_path, index=False)
    return ranked_df[FINAL_COLUMNS]


def resumemain(
    resume_directory: str,
    job_description_path: str = None,
    weights: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, bytes]]:
    """
    Main function to process and rank resumes.

    Args:
        resume_directory: Path to zip file containing PDF resumes
        job_description_path: Optional path to job description text file
        weights: Optional dictionary of scoring weights and thresholds

    Returns:
        Tuple of (DataFrame with ranked resumes and their scores, PDF cache dict)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Load and process resumes
    df, pdf_cache = load_and_process_resumes(resume_directory, weights)

    # Load job description if provided
    job_description = None
    if job_description_path:
        with open(job_description_path, "r", encoding="utf-8") as f:
            job_description = f.read()

    # Calculate all scores
    df = calculate_all_scores(df, job_description, weights)

    # Rank and export
    ranked_df = rank_and_export(df)

    # Rename columns to human-readable format for display
    display_df = ranked_df.rename(columns=DISPLAY_COLUMN_NAMES)

    return display_df, pdf_cache


def main():
    resume_directory = os.path.join(os.getcwd(), "extracted_text_files")
    resumemain(resume_directory)


if __name__ == "__main__":
    main()
