# import streamlit as st
# import tempfile
# import os
# from model import resumemain
# import zipfile
# import base64


# st.title('Resume Ranking Application')

# # uploaded_file = st.file_uploader(
# #     "Upload a zip file containing resume files",
# #     type=['zip']
# # )

# # job_description_file = st.file_uploader(
# #     "Upload a job description text file (optional)",
# #     type=['txt']
# # )

# # File upload section
# with st.container():
#     st.subheader("Upload Files")

#     # File uploader for resumes (zip file)
#     uploaded_file = st.file_uploader(
#         "Upload a zip file containing resume files",
#         type=['zip'],
#         key="resume_uploader"
#     )

#     # File uploader for optional job description (text file)
#     job_description_file = st.file_uploader(
#         "Upload a job description text file (optional)",
#         type=['txt'],
#         key="jd_uploader"
#     )

# # Display uploaded file names (Maintains space after upload)
# if uploaded_file:
#     st.success(f"✅ Resume file uploaded: {uploaded_file.name}")

# if job_description_file:
#     st.success(f"✅ Job description uploaded: {job_description_file.name}")

# output_folder = "output"

# if st.button("Process Resumes"):
#     if uploaded_file is not None:
#         print("hi, we detect that a file is uploaded by you...")
#         # Create a temporary directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Path to save the uploaded zip file
#             zip_path = os.path.join(temp_dir, uploaded_file.name)
#             print(zip_path)
            
#             # Save the uploaded zip file to the temporary directory
#             with open(zip_path, 'wb') as f:
#                 f.write(uploaded_file.getvalue())
            
#             # Save job description if provided
#             job_description_path = None
#             if job_description_file:
#                 job_description_path = os.path.join(temp_dir, job_description_file.name)
#                 with open(job_description_path, 'wb') as f:
#                     f.write(job_description_file.getvalue())

#             # Process the zip file using the function from model.py
#             with st.spinner('Processing resumes...'):
#                 print("hi, starting 2...")
#                 print(zip_path)
#                 results = resumemain(zip_path, job_description_path)

            
#             # Display each PDF resume on the webpage
#             for file_name in os.listdir(output_folder):
#                 file_path = os.path.join(output_folder, file_name)
#                 if file_name.lower().endswith('.txt'):
#                     st.write(f"### {file_name}")
#                     with open(file_path, 'rb') as pdf_file:
#                         pdf_data = pdf_file.read()
#                         st.download_button(
#                             label=f"Download {file_name}",
#                             data=pdf_data,
#                             file_name=file_name,
#                             mime="application/pdf"
#                         )
#                         st.markdown(
#                             f'<iframe src="data:application/pdf;base64,{pdf_data.encode("base64")}" width="700" height="500"></iframe>',
#                             unsafe_allow_html=True
#                         )

#             # Display the results
#             st.success('Resumes processed successfully!')
#             st.write('### Results:')
#             st.dataframe(results)
            
#             # Provide option to download the results as CSV
#             csv = results.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="Download results as CSV",
#                 data=csv,
#                 file_name='resume_results.csv',
#                 mime='text/csv',
#             )

#             # Extract top 3 resume IDs
#             top_3_ids = results['ID'].head(3).tolist()

#             # Extract PDFs from the uploaded zip file
#             extracted_pdfs = {}
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(temp_dir)
#                 for file_name in zip_ref.namelist():
#                     base_name = os.path.splitext(os.path.basename(file_name))[0]
#                     # Remove the 'candidate_' prefix to get just the ID
#                     file_id = base_name.replace("candidate_", "")
#                     if file_id in top_3_ids and file_name.lower().endswith('.pdf'):
#                         extracted_pdfs[file_id] = os.path.join(temp_dir, file_name)

#             # Display download buttons for top 3 resumes
#             st.write("### Top 3 Resumes (PDFs)")
#             for resume_id, pdf_path in extracted_pdfs.items():
#                 with open(pdf_path, "rb") as pdf_file:
#                     pdf_data = pdf_file.read()
#                     st.download_button(
#                         label=f"Download Resume {resume_id}",
#                         data=pdf_data,
#                         file_name=f"candidate_{resume_id}.pdf",
#                         mime="application/pdf"
#                     )

#                     # Embed PDF Viewer
#                     base64_pdf = base64.b64encode(pdf_data).decode()
#                     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500"></iframe>'
#                     st.markdown(pdf_display, unsafe_allow_html=True)
#     else:
#         st.write('Please upload a zip file containing resume files.')

# st.markdown("""
#     ---
#     Made with ❤️ by Samudraneel Sarkar [LinkedIn](https://www.linkedin.com/in/samudraneel-sarkar) | [GitHub](https://github.com/samudraneel05)
#      &   Guransh Goyal [LinkedIn](https://www.linkedin.com/in/guransh-goyal) | [GitHub](https://github.com/GuranshGoyal)
# """)

# st.markdown("""
#     © 2025 P-125, Batch of 2027. All rights reserved.
# """)


import streamlit as st
import tempfile
import os
from model import resumemain
import zipfile
import base64

st.title('Resume Ranking Application')

st.markdown("""
    <style>
    .description-box {
        width: 80%;
        padding: 10px;
        /* background-color: #24292e;  GitHub dark grey */
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        transition: all 0.3s ease-in-out;
        color: #4CAF50; /* Green text */
    }

    .description-box:hover {
        white-space: normal;
        overflow: visible;
    }
    </style>

    <div class="description-box">
        Hi, Recruiters! 👋
        <br>This app helps recruiters and hiring managers efficiently rank resumes based on their relevance to a job description. 
        <br>Simply upload a ZIP file with multiple resumes, optionally add a job description file (In this case, evaluation will be done entirely based on pre-defined parameters), and let the app analyze and rank the resumes for you. 🚀    </div>
""", unsafe_allow_html=True)

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
    st.success(f"✅ Resume file uploaded: {uploaded_file.name}")

if job_description_file:
    st.success(f"✅ Job description uploaded: {job_description_file.name}")

output_folder = "output"

if st.button("Process Resumes"):
    if uploaded_file is not None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path to save the uploaded zip file
            zip_path = os.path.join(temp_dir, uploaded_file.name)
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Save job description if provided
            job_description_path = None
            if job_description_file:
                job_description_path = os.path.join(temp_dir, job_description_file.name)
                with open(job_description_path, 'wb') as f:
                    f.write(job_description_file.getvalue())

            # Process the zip file using the function from model.py
            with st.spinner('Processing resumes...'):
                results = resumemain(zip_path, job_description_path)

            # Check if results are available
            if results is not None and not results.empty:
                st.success('Resumes processed successfully!')
                st.write('### Results:')
                st.dataframe(results)

                # Provide option to download the results as CSV
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='resume_results.csv',
                    mime='text/csv',
                )

                # Extract top 3 resume IDs
                # top_3_ids = results['ID'].head(3).tolist()
                top_3_ids = [str(i).zfill(3) for i in results['ID'].head(3).tolist()]

                # Extract PDFs from the uploaded zip file
                extracted_pdfs = {}
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    for file_name in zip_ref.namelist():
                        base_name = os.path.splitext(os.path.basename(file_name))[0]
                        # Remove the 'candidate_' prefix to get just the ID
                        file_id = base_name.replace("candidate_", "")
                        if file_id in top_3_ids and file_name.lower().endswith('.pdf'):
                            extracted_pdfs[file_id] = os.path.join(temp_dir, file_name)
                
                # Sort the extracted PDFs based on top_3_ids (so the highest-ranked comes first)
                sorted_pdfs = sorted(extracted_pdfs.items(), key=lambda x: top_3_ids.index(x[0]))

                # Display download buttons for top 3 resumes
                st.write("### Top 3 Resumes (PDFs)")
                if sorted_pdfs:
                    for resume_id, pdf_path in sorted_pdfs:
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                            st.download_button(
                                label=f"Download Resume {resume_id}",
                                data=pdf_data,
                                file_name=f"candidate_{resume_id}.pdf",
                                mime="application/pdf"
                            )

                            # Embed PDF Viewer
                            base64_pdf = base64.b64encode(pdf_data).decode()
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.warning("No top resumes found in the zip file.")

            else:
                st.error("No valid results from the resume processing.")
    else:
        st.write('Please upload a zip file containing resume files.')

# st.markdown("""
#     ---
#     Made with ❤️ by Samudraneel Sarkar [LinkedIn](https://www.linkedin.com/in/samudraneel-sarkar) | [GitHub](https://github.com/samudraneel05)
#      &   Guransh Goyal [LinkedIn](https://www.linkedin.com/in/guransh-goyal) | [GitHub](https://github.com/GuranshGoyal)
# """)

# st.markdown("""
#     © 2025 P-125, Batch of 2027. All rights reserved.
# """)
# Credits section with a beautiful layout and email addresses
# Credits section with a beautiful layout and email addresses
# Credits section with a beautiful layout, side-by-side display and custom hyperlink color
# Credits section with a beautiful layout, side-by-side display and custom hyperlink color
import streamlit as st

import streamlit as st

import streamlit as st

st.markdown("""
    ---
    <div style="text-align: center; font-size: 24px; font-weight: bold;">
        -- Made with ❤️ by --
    </div>
    <div style="display: flex; justify-content: center; gap: 50px; font-size: 23px;">
        <div style="text-align: center;">
            <p style="font-size: 20px;">Samudraneel Sarkar</p>
            <p style="font-size: 16px;">
                <a href="https://www.linkedin.com/in/samudraneel-sarkar" target="_blank" style="color: #0077b5; text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/samudraneel05" target="_blank" style="color: #333; text-decoration: none;">GitHub</a>
            </p>
            <p style="font-size: 16px;">📧 <a href="mailto:samudraneel05@gmail.com" style="color: #FFFFFF; text-decoration: none;">samudraneel05@gmail.com</a></p>
        </div>
        <div style="text-align: center;">
            <p style="font-size: 20px;">Guransh Goyal</p>
            <p style="font-size: 16px;">
                <a href="https://www.linkedin.com/in/guransh-goyal" target="_blank" style="color: #0077b5; text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/GuranshGoyal" target="_blank" style="color: #333; text-decoration: none;">GitHub</a>
            </p>
            <p style="font-size: 16px;">📧 <a href="mailto:guransh31goyal@gmail.com" style="color: #FFFFFF; text-decoration: none;">guransh31goyal@gmail.com</a></p>
        </div>
    </div>
    <div style="text-align: center; font-size: 14px; color: gray;">
        <p>© 2025 P-125, Batch of 2027. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
