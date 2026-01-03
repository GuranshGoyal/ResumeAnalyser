import streamlit as st
import tempfile
import os
from model import resumemain, DEFAULT_WEIGHTS
import base64

st.set_page_config(page_title="Aplytic - Resume Ranking", layout="wide")

st.title('Aplytic - Resume Ranking Application')

st.markdown("""
    <style>
    .description-box {
        width: 80%;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        transition: all 0.3s ease-in-out;
        color: #4CAF50;
    }

    .description-box:hover {
        white-space: normal;
        overflow: visible;
    }
    </style>

<div>Simply upload a ZIP file with multiple resumes, optionally add a job description file, and let the app analyze and rank the resumes for you.</div>
""", unsafe_allow_html=True)


# Sidebar for weight configuration
with st.sidebar:
    st.header("Scoring Configuration")
    st.markdown("Adjust weights and thresholds to customize scoring")

    # Initialize weights in session state if not present
    if 'weights' not in st.session_state:
        st.session_state.weights = DEFAULT_WEIGHTS.copy()

    # Reset button
    if st.button("Reset to Defaults"):
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
        st.rerun()

    st.divider()

    # Final Score Weights
    with st.expander("Final Score Weights", expanded=True):
        st.markdown("*How much overall score vs job match matters*")

        col1, col2 = st.columns(2)
        with col1:
            overall_weight = st.slider(
                "Overall Score Weight",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights["final_overall_weight"],
                step=0.05,
                key="final_overall_weight",
                help="Weight for overall resume quality score"
            )
        with col2:
            tfidf_weight = st.slider(
                "Job Match (TF-IDF) Weight",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights["final_tfidf_weight"],
                step=0.05,
                key="final_tfidf_weight",
                help="Weight for job description matching score"
            )

        # Normalize to ensure they sum to 1
        total = overall_weight + tfidf_weight
        if total > 0:
            st.session_state.weights["final_overall_weight"] = overall_weight / total
            st.session_state.weights["final_tfidf_weight"] = tfidf_weight / total
        st.caption(f"Normalized: Overall={st.session_state.weights['final_overall_weight']:.2f}, Job Match={st.session_state.weights['final_tfidf_weight']:.2f}")

    # Overall Score Weights
    with st.expander("Overall Score Weights", expanded=False):
        st.markdown("*Balance between technical, managerial, and resume quality*")

        tech_w = st.slider(
            "Technical Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["overall_technical_weight"],
            step=0.05,
            key="overall_technical_weight",
            help="Importance of technical skills and experience"
        )
        mgr_w = st.slider(
            "Managerial Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["overall_managerial_weight"],
            step=0.05,
            key="overall_managerial_weight",
            help="Importance of leadership and soft skills"
        )
        qual_w = st.slider(
            "Resume Quality Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["overall_quality_weight"],
            step=0.05,
            key="overall_quality_weight",
            help="Importance of resume formatting and writing quality"
        )

        # Normalize
        total = tech_w + mgr_w + qual_w
        if total > 0:
            st.session_state.weights["overall_technical_weight"] = tech_w / total
            st.session_state.weights["overall_managerial_weight"] = mgr_w / total
            st.session_state.weights["overall_quality_weight"] = qual_w / total
        st.caption(f"Normalized: Tech={st.session_state.weights['overall_technical_weight']:.2f}, Mgr={st.session_state.weights['overall_managerial_weight']:.2f}, Quality={st.session_state.weights['overall_quality_weight']:.2f}")

    # Technical Score Weights
    with st.expander("Technical Score Weights", expanded=False):
        st.markdown("*Components of technical evaluation*")

        skills_w = st.slider(
            "Skills Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["technical_skills_weight"],
            step=0.05,
            key="technical_skills_weight",
            help="Importance of number of relevant skills"
        )
        exp_w = st.slider(
            "Experience Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["technical_experience_weight"],
            step=0.05,
            key="technical_experience_weight",
            help="Importance of years of experience"
        )
        edu_w = st.slider(
            "Education Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["technical_education_weight"],
            step=0.05,
            key="technical_education_weight",
            help="Importance of education level"
        )

        # Normalize
        total = skills_w + exp_w + edu_w
        if total > 0:
            st.session_state.weights["technical_skills_weight"] = skills_w / total
            st.session_state.weights["technical_experience_weight"] = exp_w / total
            st.session_state.weights["technical_education_weight"] = edu_w / total
        st.caption(f"Normalized: Skills={st.session_state.weights['technical_skills_weight']:.2f}, Exp={st.session_state.weights['technical_experience_weight']:.2f}, Edu={st.session_state.weights['technical_education_weight']:.2f}")

    # Managerial Score Weights
    with st.expander("Managerial Score Weights", expanded=False):
        st.markdown("*Components of managerial/soft skills evaluation*")

        soft_w = st.slider(
            "Soft Skills Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["managerial_soft_skills_weight"],
            step=0.05,
            key="managerial_soft_skills_weight",
            help="Importance of communication and interpersonal skills"
        )
        achieve_w = st.slider(
            "Achievement Impact Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["managerial_achievement_weight"],
            step=0.05,
            key="managerial_achievement_weight",
            help="Importance of quantified achievements"
        )
        lead_w = st.slider(
            "Leadership Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["managerial_leadership_weight"],
            step=0.05,
            key="managerial_leadership_weight",
            help="Importance of leadership experience"
        )

        # Normalize
        total = soft_w + achieve_w + lead_w
        if total > 0:
            st.session_state.weights["managerial_soft_skills_weight"] = soft_w / total
            st.session_state.weights["managerial_achievement_weight"] = achieve_w / total
            st.session_state.weights["managerial_leadership_weight"] = lead_w / total
        st.caption(f"Normalized: Soft={st.session_state.weights['managerial_soft_skills_weight']:.2f}, Achieve={st.session_state.weights['managerial_achievement_weight']:.2f}, Lead={st.session_state.weights['managerial_leadership_weight']:.2f}")

    # Thresholds and Caps
    with st.expander("Thresholds & Caps", expanded=False):
        st.markdown("*Scoring limits and boundaries*")

        st.session_state.weights["skill_cap"] = st.number_input(
            "Skill Count Cap",
            min_value=1,
            max_value=50,
            value=int(st.session_state.weights["skill_cap"]),
            step=1,
            help="Maximum number of skills to count (diminishing returns beyond this)"
        )

        st.session_state.weights["experience_cap_technical"] = st.number_input(
            "Technical Experience Cap (years)",
            min_value=1,
            max_value=40,
            value=int(st.session_state.weights["experience_cap_technical"]),
            step=1,
            help="Years of experience that gives maximum technical score"
        )

        st.session_state.weights["experience_cap_leadership"] = st.number_input(
            "Leadership Experience Cap (years)",
            min_value=1,
            max_value=40,
            value=int(st.session_state.weights["experience_cap_leadership"]),
            step=1,
            help="Years of experience that gives maximum leadership score"
        )

    # Brevity Settings
    with st.expander("Resume Length (Brevity) Settings", expanded=False):
        st.markdown("*Optimal resume word count parameters*")

        st.session_state.weights["brevity_min_words"] = st.number_input(
            "Minimum Word Count",
            min_value=50,
            max_value=500,
            value=int(st.session_state.weights["brevity_min_words"]),
            step=25,
            help="Resumes below this are penalized as too short"
        )

        st.session_state.weights["brevity_optimal_words"] = st.number_input(
            "Optimal Word Count",
            min_value=200,
            max_value=1500,
            value=int(st.session_state.weights["brevity_optimal_words"]),
            step=50,
            help="Ideal resume length"
        )

        st.session_state.weights["brevity_max_words"] = st.number_input(
            "Maximum Word Count",
            min_value=500,
            max_value=3000,
            value=int(st.session_state.weights["brevity_max_words"]),
            step=100,
            help="Resumes above this are penalized as too long"
        )

    # Education Level Scores
    with st.expander("Education Level Scores", expanded=False):
        st.markdown("*Score assigned to each education level*")

        st.session_state.weights["education_phd"] = st.slider(
            "PhD Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["education_phd"],
            step=0.05,
            key="education_phd"
        )

        st.session_state.weights["education_postgraduate"] = st.slider(
            "Postgraduate/Master's Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["education_postgraduate"],
            step=0.05,
            key="education_postgraduate"
        )

        st.session_state.weights["education_bachelor"] = st.slider(
            "Bachelor's Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["education_bachelor"],
            step=0.05,
            key="education_bachelor"
        )

        st.session_state.weights["education_associate"] = st.slider(
            "Associate's Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["education_associate"],
            step=0.05,
            key="education_associate"
        )

        st.session_state.weights["education_other"] = st.slider(
            "Other/Unknown Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights["education_other"],
            step=0.05,
            key="education_other"
        )

    st.divider()

    # Display current weights summary
    with st.expander("Current Configuration Summary", expanded=False):
        st.json(st.session_state.weights)


# Main content area
with st.container():
    st.subheader("Upload Files")

    uploaded_file = st.file_uploader(
        "Upload a zip file containing resume files",
        type=['zip'],
        key="resume_uploader"
    )

    job_description_file = st.file_uploader(
        "Upload a job description text file (optional)",
        type=['txt'],
        key="jd_uploader"
    )

if uploaded_file:
    st.success(f"Resume file uploaded: {uploaded_file.name}")

if job_description_file:
    st.success(f"Job description uploaded: {job_description_file.name}")

if st.button("Process Resumes", type="primary"):
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded zip file
            zip_path = os.path.join(temp_dir, uploaded_file.name)
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Save job description if provided
            job_description_path = None
            if job_description_file:
                job_description_path = os.path.join(temp_dir, job_description_file.name)
                with open(job_description_path, 'wb') as f:
                    f.write(job_description_file.getvalue())

            # Process resumes and get PDF cache
            with st.spinner('Processing resumes with your custom scoring configuration...'):
                results, pdf_cache = resumemain(zip_path, job_description_path, st.session_state.weights)

            # Check if results are available
            if results is not None and not results.empty:
                st.success('Resumes processed successfully!')

                # Display weight configuration used
                with st.expander("Scoring Configuration Used", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Final Score**")
                        st.write(f"Overall: {st.session_state.weights['final_overall_weight']:.0%}")
                        st.write(f"Job Match: {st.session_state.weights['final_tfidf_weight']:.0%}")
                    with col2:
                        st.markdown("**Overall Score**")
                        st.write(f"Technical: {st.session_state.weights['overall_technical_weight']:.0%}")
                        st.write(f"Managerial: {st.session_state.weights['overall_managerial_weight']:.0%}")
                        st.write(f"Quality: {st.session_state.weights['overall_quality_weight']:.0%}")
                    with col3:
                        st.markdown("**Caps**")
                        st.write(f"Skills: {st.session_state.weights['skill_cap']}")
                        st.write(f"Tech Exp: {st.session_state.weights['experience_cap_technical']} yrs")
                        st.write(f"Lead Exp: {st.session_state.weights['experience_cap_leadership']} yrs")

                st.write('### Results:')
                st.dataframe(results, use_container_width=True)

                # Provide option to download the results as CSV
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='resume_results.csv',
                    mime='text/csv',
                )

                # Get top 3 resume IDs (normalized to unpadded string)
                top_3_ids = [str(i) for i in results['ID'].head(3).tolist()]

                # Display top 3 resumes using cached PDF data
                st.write("### Top 3 Resumes (PDFs)")

                displayed_count = 0
                for resume_id in top_3_ids:
                    if resume_id in pdf_cache:
                        pdf_data = pdf_cache[resume_id]
                        displayed_count += 1

                        st.download_button(
                            label=f"Download Resume {resume_id}",
                            data=pdf_data,
                            file_name=f"candidate_{resume_id}.pdf",
                            mime="application/pdf",
                            key=f"download_{resume_id}"
                        )

                        # Embed PDF Viewer
                        base64_pdf = base64.b64encode(pdf_data).decode()
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)

                if displayed_count == 0:
                    st.warning("No top resumes found in the cache.")

            else:
                st.error("No valid results from the resume processing.")
    else:
        st.write('Please upload a zip file containing resume files.')


st.markdown("""
    ---
    <div style="text-align: center; font-size: 24px; font-weight: bold;">
        -- Made with love by --
    </div>
    <div style="display: flex; justify-content: center; gap: 50px; font-size: 23px;">
        <div style="text-align: center;">
            <p style="font-size: 20px;">Samudraneel Sarkar</p>
            <p style="font-size: 16px;">
                <a href="https://www.linkedin.com/in/samudraneel-sarkar" target="_blank" style="color: #0077b5; text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/samudraneel05" target="_blank" style="color: #333; text-decoration: none;">GitHub</a>
            </p>
            <p style="font-size: 16px;"><a href="mailto:samudraneel05@gmail.com" style="color: #FFFFFF; text-decoration: none;">samudraneel05@gmail.com</a></p>
        </div>
        <div style="text-align: center;">
            <p style="font-size: 20px;">Guransh Goyal</p>
            <p style="font-size: 16px;">
                <a href="https://www.linkedin.com/in/guransh-goyal" target="_blank" style="color: #0077b5; text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/GuranshGoyal" target="_blank" style="color: #333; text-decoration: none;">GitHub</a>
            </p>
            <p style="font-size: 16px;"><a href="mailto:guransh31goyal@gmail.com" style="color: #FFFFFF; text-decoration: none;">guransh31goyal@gmail.com</a></p>
        </div>
    </div>
    <div style="text-align: center; font-size: 14px; color: gray;">
        <p>2025 P-125, Batch of 2027. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
