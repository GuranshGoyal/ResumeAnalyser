#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymupdf

# import pathlib
import zipfile
import os
import re
import pandas as pd
import spacy
import language_tool_python
import datetime
import json
from typing import List, Dict

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import tempfile
from PyPDF2 import PdfReader
from io import BytesIO

# In[4]:


# fileName=input()

# with pymupdf.open(fileName) as doc:
#     text=chr(12).join([page.get_text() for page in doc])

# pathlib.Path(fileName+".txt").write_bytes(text.encode())


# In[5]:


def extract_text_from_pdf(pdf_path):
    with pymupdf.open(pdf_path) as pdf:
        text = ""
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
        return text


# In[6]:


# # Function to extract PDFs from zipfile, convert them to text, and save as .txt files
# def extract_and_save_text_from_zip(zip_file_path, extract_to, output_zip_path):
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)

#     # Create a new zip file to store the .txt files
#     with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
#         for root, dirs, files in os.walk(extract_to):
#             for file in files:
#                 if file.endswith(".pdf"):
#                     pdf_path = os.path.join(root, file)
#                     text = extract_text_from_pdf(pdf_path)

#                     # Save the extracted text as a .txt file
#                     text_filename = os.path.splitext(file)[0] + ".txt"
#                     text_filepath = os.path.join(extract_to, text_filename)
#                     with open(text_filepath, 'w', encoding='utf-8') as text_file:
#                         text_file.write(text)

#                     # Add the text file to the output zip
#                     output_zip.write(text_filepath, arcname=text_filename)

# # Example usage
# zip_file_path = "Final_Resumes.zip"  # Path to your input zipfile containing PDFs
# extract_to = "Final_Resumes_Text"  # Temporary directory to extract PDFs
# output_zip_path = "extracted_text_files.zip"  # Path to the output zipfile with text files

# # Extract text and create a zip of .txt files
# extract_and_save_text_from_zip(zip_file_path, extract_to, output_zip_path)

# print(f"Text files saved and zipped as {output_zip_path}")


# In[ ]:


def extract_text_from_pdf_zip(zip_file, output_folder=None):
    """
    Extracts text from PDF files within a zip archive and saves them as text files.

    Args:
        zip_file (str or bytes or file-like object): Path to the zip file containing PDF files,
            or a bytes object, or a file-like object.
        output_folder (str, optional): Path to the folder where the extracted text files will be saved.
            If None, a temporary directory will be created.

    Returns:
        str: Path to the output folder containing the extracted text files.
    """
    # Determine if zip_file is a path, bytes object, or file-like object
    if isinstance(zip_file, str):
        # It's a file path
        zip_file_path = zip_file
    elif isinstance(zip_file, bytes):
        # It's a bytes object
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file)
        temp_zip.close()
        zip_file_path = temp_zip.name
    elif hasattr(zip_file, "read"):
        # It's a file-like object (e.g., uploaded file)
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file.read())
        temp_zip.close()
        zip_file_path = temp_zip.name
    else:
        raise ValueError(
            "zip_file must be a file path, bytes object, or file-like object."
        )

    # Create the output folder if not provided
    if output_folder is None:
        output_folder = tempfile.mkdtemp()
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Get list of PDF files
        pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith(".pdf")]
        if not pdf_files:
            raise ValueError("No PDF files found in the zip archive.")

        # Print the list of recognized PDF files
        print("PDF files recognized in the zip archive:")
        for pdf_file in pdf_files:
            print(pdf_file)

        # Process each PDF file
        for pdf_file_name in pdf_files:
            # Extract PDF file to a temporary location
            with zip_ref.open(pdf_file_name) as pdf_file:
                # Read PDF content
                # Wrap the file stream in BytesIO to make it binary
                pdf_stream = BytesIO(pdf_file.read())
                pdf_reader = PdfReader(pdf_stream)

                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                # Extract the ID from the PDF filename
                base_name = os.path.splitext(os.path.basename(pdf_file_name))[0]
                match = re.match(r"candidate_(\d+)", base_name, re.IGNORECASE)
                if match:
                    candidate_id = match.group(1)
                    # Create the text filename in the desired format
                    text_file_name = f"Resume_of_ID_{candidate_id}.txt"
                else:
                    # If the filename doesn't match the expected pattern, use a default name
                    text_file_name = f"{base_name}.txt"

                # Construct the full path to the text file in the output folder
                text_file_path = os.path.join(output_folder, text_file_name)

                # Save the extracted text to the text file
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)

    # Clean up temporary zip file if one was created
    if not isinstance(zip_file, str):
        os.remove(zip_file_path)

    # Print the names of the text files in the output folder
    print("\nText files saved in the output folder:")
    for file_name in os.listdir(output_folder):
        print(file_name)

    return output_folder


# In[ ]:


def load_and_clean_data(directory):
    """Conversion of zip of pdfs to folder of extracted text files"""
    text_files_folder = extract_text_from_pdf_zip(directory)
    print(text_files_folder)
    print("Now we have a folder of text files")

    """Load data from text files in the specified directory and perform initial cleaning."""
    data = []
    for filename in os.listdir(text_files_folder):
        if filename.startswith("Resume_of_ID_") and filename.endswith(".txt"):
            file_path = os.path.join(text_files_folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            id_number = int(filename.split("_")[3].split(".")[0])
            # id_number = int(filename.split('_')[1].split('.')[0])
            clean_text = " ".join(text.split())  # Remove extra whitespace
            data.append({"ID": id_number, "Text": clean_text})
    return pd.DataFrame(data)


# In[8]:


def preprocess_text(text):
    """Preprocess text using spaCy for tokenization and lemmatization."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens


# In[9]:


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


# In[10]:


def detect_education_level(text):
    """Detect the highest education level mentioned in the resume."""
    education_patterns = {
        "PhD": r"\bPh\.?D\.?\b|\bDoctor(ate)?\b",
        # 'Master': r'\bM\.?S\.?\b|\bM\.?A\.?\b|\bM\.?Tech\b|\bMaster(s)?\b',
        "Postgraduate": r"\bM\.?S\.?\b|\bM\.?A\.?\b|\bM\.?Tech\b|\bM\.?Sc\b|\bMaster(s)?\b|\bPost\s?Graduation\b|\bPostgraduate\b",
        "Bachelor": r"\bB\.?S\.?\b|\bB\.?A\.?\b|\bB\.?Tech\b|\bB\.?Sc\b|\bBachelor(s)?\b",
        # 'Associate': r'\bA\.?S\.?\b|\bA\.?A\.?\b|\bAssociate\b'
    }

    for level, pattern in education_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return "Other"


# In[11]:


def calculate_spell_check_ratio(text):
    """Calculate the ratio of potential spelling errors to total words."""
    matches = language_tool_python.LanguageToolPublicAPI("en-US").check(text)
    total_words = len(text.split())
    return 1 - (len(matches) / total_words)


# In[12]:


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


# In[13]:


def quantify_brevity(text):
    """Quantify the brevity of the resume."""
    word_count = len(text.split())
    if word_count < 200:
        return 0.5  # Too short
    elif word_count > 1000:
        return 0.5  # Too long
    else:
        return 1 - (abs(600 - word_count) / 400)  # Optimal around 600 words


# In[14]:


# def process_resume(row):
#     """Process a single resume and return a dictionary of features."""
#     text = row['Text']
#     tokens = preprocess_text(text)

#     years_of_experience = extract_years_of_experience(text)
#     education_level = detect_education_level(text)
#     spell_check_ratio = calculate_spell_check_ratio(text)
#     section_score = identify_resume_sections(text)
#     brevity_score = quantify_brevity(text)
#     extracted_skills=extract_skills(text)

#     # Generate job description based on the resume
#     job_description = generate_job_description(text)

#     return {
#         'ID': row['ID'],
#         'Preprocessed_Tokens': tokens,
#         'Years_of_Experience': years_of_experience,
#         'Education_Level': education_level,
#         'Spell_Check_Ratio': spell_check_ratio,
#         'Section_Score': section_score,
#         'Brevity_Score': brevity_score,
#         'Extracted_skills':extracted_skills
#         'Generated_Job_Description': job_description
#     }


# In[15]:


def calculate_word_sentence_counts(text):
    """Calculate word count and sentence count."""
    sentences = re.split(r"[.!?]+", text)
    word_count = len(text.split())
    sentence_count = len([s for s in sentences if s.strip()])
    return word_count, sentence_count


# In[17]:


def calculate_skill_match_score(resume_skills, job_skills):
    """Calculate the skill match score."""
    if not job_skills:
        return 0
    matched_skills = set(resume_skills) & set(job_skills)
    return len(matched_skills) / len(job_skills)


# In[18]:


def analyze_sentiment(text):
    """Analyze the sentiment of achievement statements in the resume."""
    blob = TextBlob(text)
    return blob.sentiment.polarity


# In[19]:


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
            impact_score += (
                float(value) / 1000
            )  # Assume larger numbers for non-percentage values
    return min(impact_score, 1)


# In[20]:


def calculate_technical_score(row):
    """Calculate the technical CV score."""
    skill_count = len(row["Extracted_Skills"])
    experience_score = min(row["Years_of_Experience"] / 10, 1)  # Cap at 10 years
    education_score = {
        "PhD": 1,
        "Master": 0.8,
        "Bachelor": 0.6,
        "Associate": 0.4,
        "Other": 0.2,
    }.get(row["Education_Level"], 0.2)

    return skill_count / 10 * 0.4 + experience_score * 0.3 + education_score * 0.3


# In[21]:


def calculate_managerial_score(row):
    """Calculate the managerial CV score."""
    soft_skills_score = analyze_sentiment(row["Text"])
    achievement_impact = quantify_achievement_impact(row["Text"])
    leadership_score = min(
        row["Years_of_Experience"] / 15, 1
    )  # Assume leadership potential increases with experience

    return soft_skills_score * 0.3 + achievement_impact * 0.4 + leadership_score * 0.3


# In[22]:


def calculate_overall_score(row):
    """Calculate the overall CV score."""
    technical_score = row["Technical_Score"]
    managerial_score = row["Managerial_Score"]
    resume_quality_score = (
        row["Spell_Check_Ratio"] + row["Section_Score"] + row["Brevity_Score"]
    ) / 3
    return technical_score * 0.4 + managerial_score * 0.3 + resume_quality_score * 0.3


# In[23]:


def process_resume_section2(row, job_skills):
    """Process a single resume for Section 2 analysis."""
    word_count, sentence_count = calculate_word_sentence_counts(row["Text"])

    # Combine general and technical skills
    all_skills = job_skills["general_skills"] + job_skills["technical_skills"]

    # Extract skills based on the generated job description
    job_specific_skills = extract_skills(row["Generated_Job_Description"], all_skills)
    resume_skills = extract_skills(row["Text"], all_skills)

    technical_score = calculate_technical_score(row, job_specific_skills)
    managerial_score = calculate_managerial_score(row)
    resume_quality_score = (
        row["Spell_Check_Ratio"] + row["Section_Score"] + row["Brevity_Score"]
    ) / 3
    overall_score = calculate_overall_score(
        technical_score, managerial_score, resume_quality_score
    )

    return {
        "ID": row["ID"],
        "Word_Count": word_count,
        "Sentence_Count": sentence_count,
        "Resume_Skills": resume_skills,
        "Job_Specific_Skills": job_specific_skills,
        "Technical_Score": technical_score,
        "Managerial_Score": managerial_score,
        "Resume_Quality_Score": resume_quality_score,
        "Overall_Score": overall_score,
    }


# In[24]:


def normalize_scores(df):
    """Normalize scores to ensure fair comparison across all resumes."""
    score_columns = [
        "Technical_Score",
        "Managerial_Score",
        "Resume_Quality_Score",
        "Overall_Score",
    ]
    for column in score_columns:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )
    return df


# In[25]:


def process_resumes_section2(df, job_skills):
    """Process all resumes for Section 2 analysis."""
    results = []
    for _, row in df.iterrows():
        results.append(process_resume_section2(row, job_skills))

    results_df = pd.DataFrame(results)
    normalized_df = normalize_scores(results_df)
    return normalized_df


# In[26]:


# This function will be called from the main function in Section 3
def run_section2(input_file, job_skills):
    """Run Section 2 processing on the input file."""
    df = pd.read_csv(input_file)
    processed_df = process_resumes_section2(df, job_skills)
    processed_df.to_csv("processed_resumes_section2.csv", index=False)
    print(
        "Section 2 processing completed. Results saved to 'processed_resumes_section2.csv'"
    )
    return processed_df


# In[ ]:


# In[28]:


# def load_job_skills(file_path: str) -> List[str]:
#     """Load job skills from a JSON file."""
#     with open(file_path, "r") as file:
#         return json.load(file)


# In[29]:


def job_description_matching(resume_text: str, job_description: str) -> float:
    """Calculate similarity between resume and job description."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


# In[30]:


def adjust_scores_with_job_match(
    df: pd.DataFrame, job_description: str
) -> pd.DataFrame:
    """Adjust scores based on job description matching."""
    df["Job_Match_Score"] = df["Text"].apply(
        lambda x: job_description_matching(x, job_description)
    )
    df["Adjusted_Overall_Score"] = (
        df["Overall_Score"] * 0.7 + df["Job_Match_Score"] * 0.3
    )
    return df


# In[31]:


def rank_resumes(df: pd.DataFrame) -> pd.DataFrame:
    """Rank resumes based on adjusted overall score."""
    return df.sort_values("Adjusted_Overall_Score", ascending=False).reset_index(
        drop=True
    )


# In[32]:


def generate_report(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """Generate a report with top candidates and summary statistics."""
    top_candidates = df.head(top_n)[
        [
            "ID",
            "Adjusted_Overall_Score",
            "Technical_Score",
            "Managerial_Score",
            "Job_Match_Score",
        ]
    ]
    summary_stats = df[
        [
            "Adjusted_Overall_Score",
            "Technical_Score",
            "Managerial_Score",
            "Job_Match_Score",
        ]
    ].describe()

    return {
        "top_candidates": top_candidates.to_dict(orient="records"),
        "summary_stats": summary_stats.to_dict(),
    }


# In[33]:


def match_resume_to_job_description(resume_text, job_description):
    """Match a resume to a specific job description and return adjusted scores."""
    # Reuse the existing job_description_matching function
    match_score = job_description_matching(resume_text, job_description)

    # # Recalculate scores based on the new job description
    # job_specific_skills = extract_skills(job_description, general_job_skills)
    # resume_skills = extract_skills(resume_text, job_specific_skills)

    # technical_score = calculate_technical_score({'Text': resume_text, 'Years_of_Experience': extract_years_of_experience(resume_text), 'Education_Level': detect_education_level(resume_text)}, job_specific_skills)
    # managerial_score = calculate_managerial_score({'Text': resume_text})
    # resume_quality_score = (calculate_spell_check_ratio(resume_text) + identify_resume_sections(resume_text) + quantify_brevity(resume_text)) / 3
    # overall_score = calculate_overall_score(technical_score, managerial_score, resume_quality_score)

    # adjusted_overall_score = overall_score * 0.7 + match_score * 0.3

    return {
        # 'Technical_Score': technical_score,
        # 'Managerial_Score': managerial_score,
        # 'Resume_Quality_Score': resume_quality_score,
        # 'Overall_Score': overall_score,
        "Job_Match_Score": match_score,
        # 'Adjusted_Overall_Score': adjusted_overall_score
    }


# In[34]:


def create_default_job_skills_file(file_path):
    """Create a default job_skills.json file if it doesn't exist."""
    default_skills = {
        "general_skills": [
            "communication",
            "teamwork",
            "leadership",
            "problem-solving",
            "time management",
            "analytical skills",
            "creativity",
            "adaptability",
        ],
        "technical_skills": [
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
        ],
    }

    with open(file_path, "w") as file:
        json.dump(default_skills, file, indent=2)


# In[35]:


# In[36]:


def load_job_skills(file_path: str) -> List[str]:
    """Load general job skills from a JSON file or use a default list."""
    default_skills = [
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

    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                skills_data = json.load(file)
            if isinstance(skills_data, list):
                return skills_data
            elif isinstance(skills_data, dict) and "skills" in skills_data:
                return skills_data["skills"]
            else:
                print(f"Unexpected format in '{file_path}'. Using default skills list.")
                return default_skills
        else:
            print(f"'{file_path}' not found. Using default skills list.")
            return default_skills
    except json.JSONDecodeError:
        print(f"Error reading '{file_path}'. Using default skills list.")
        return default_skills


# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using NLP techniques."""
    if not text:
        return []

    doc = nlp(text)
    keyword_skills = set()
    ner_skills = set()

    # Match skills using keyword search
    for skill in general_skills:
        if skill.lower() in text.lower():
            keyword_skills.add(skill)

    # Extract potential skills using Named Entity Recognition (NER)
    for ent in doc.ents:
        if ent.label_ in {
            "ORG",
            "PRODUCT",
            "WORK_OF_ART",
        }:  # Some skills appear as products/orgs
            ner_skills.add(ent.text)

    # Sort each group separately
    sorted_keyword_skills = sorted(keyword_skills)
    sorted_ner_skills = sorted(ner_skills)

    # Maintain their order separately but combine them in the final output
    return sorted_keyword_skills + sorted_ner_skills  # Keywords first, NER second


# In[38]:


def process_resume(row):
    """Process a single resume and return a dictionary of features."""
    text = row["Text"]

    years_of_experience = extract_years_of_experience(text)
    education_level = detect_education_level(text)
    spell_check_ratio = calculate_spell_check_ratio(text)
    section_score = identify_resume_sections(text)
    brevity_score = quantify_brevity(text)
    extracted_skills = extract_skills(text)

    return {
        "Years_of_Experience": years_of_experience,
        "Education_Level": education_level,
        "Spell_Check_Ratio": spell_check_ratio,
        "Section_Score": section_score,
        "Brevity_Score": brevity_score,
        "Extracted_Skills": extracted_skills,
    }


# In[44]:


def resumemain(
    resume_directory: str, job_description_path: str = None
):  # resume_directory is a zip file containing resumes in pdf format
    # Adjust pandas display settings to show the full row (for debugging)
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option(
        "display.width", None
    )  # Remove line width limit'print("1. Starting resume analysis process...")

    print(resume_directory)
    # Load general skills
    print("2. Loading general skills...")
    global general_skills
    general_skills = load_job_skills("job_skills.json")

    # Load and preprocess data
    print("3. Loading and preprocessing resumes...")
    df = load_and_clean_data(
        resume_directory
    )  # zip containing pdf resumes -> folder with text form of resumes -> fit data into a dataframe
    print("0")
    print(df.iloc[0])
    print(df.head())
    print(df.columns)
    df["processed"] = df.apply(process_resume, axis=1)
    print("1")
    print(df.iloc[0])
    print(df.head())
    print(df.columns)
    df = pd.concat([df, pd.DataFrame(df["processed"].tolist())], axis=1)
    print("2")
    print(df.iloc[0])
    print(df.head())
    print(df.columns)
    df.drop("processed", axis=1, inplace=True)
    print("3")
    print(df.iloc[0])
    print(df.head())
    print(df.columns)

    # Calculate scores
    # Note: Don't drop the 'Text' even now as its required for some internal functions of the following functions
    print("4. Calculating scores...")
    df["Skill_Count"] = df["Extracted_Skills"].apply(len)
    df["Technical_Score"] = df.apply(calculate_technical_score, axis=1)
    df["Managerial_Score"] = df.apply(calculate_managerial_score, axis=1)
    df["Overall(featured)_Score"] = df.apply(
        calculate_overall_score, axis=1
    )  # this column can be added at last only (because its internal functions need the above created columns)

    if job_description_path:
        with open(job_description_path, "r", encoding="utf-8") as f:
            job_description = f.read()
    else:
        job_description = None  # Indicates no job description provided

    df["TF-IDF_Score"] = df.apply(
        lambda row: (
            match_resume_to_job_description(row["Text"], job_description).get(
                "Job_Match_Score", 1.0
            )
            if job_description
            else 1.0
        ),
        axis=1,
    )

    df["Final_Score"] = df["Overall(featured)_Score"] * 0.7 + df["TF-IDF_Score"] * 0.3
    # Rank resumes
    print("5. Ranking resumes...")
    ranked_df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    print("4")
    print(df.head())
    print(df.columns)

    # Save final results
    print("6. Saving results...")
    final_columns = [
        "ID",
        "Final_Score",
        "Overall(featured)_Score",
        "TF-IDF_Score",
        "Years_of_Experience",
        "Education_Level",
        "Technical_Score",
        "Managerial_Score",
        "Spell_Check_Ratio",
        "Section_Score",
        "Brevity_Score",
        "Skill_Count",
        "Extracted_Skills",
    ]
    ranked_df[final_columns].to_csv("final_ranked_resumes.csv", index=False)

    print("7. Resume analysis complete. Results saved to 'final_ranked_resumes.csv'")

    return ranked_df[final_columns]


# In[45]:


def main():
    # resume_directory = '/workspaces/ResumeAnalyser/extracted_text_files'
    resume_directory = os.path.join(os.getcwd(), "extracted_text_files")
    resumemain(resume_directory)


if __name__ == "__main__":
    main()


# In[ ]:


# !pip install tqdm
