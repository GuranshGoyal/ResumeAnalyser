import streamlit as st
import tempfile
import os
from model import resumemain

st.title('Resume Ranking Application')

uploaded_file = st.file_uploader(
    "Upload a zip file containing resume files",
    type=['zip']
)

output_folder = "output"

if uploaded_file is not None:
    print("hi, we detect that a file is uploaded by you...")
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Path to save the uploaded zip file
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        print(zip_path)
        
        # Save the uploaded zip file to the temporary directory
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Process the zip file using the function from model.py
        with st.spinner('Processing resumes...'):
            print("hi, starting 2...")
            print(zip_path)
            results = resumemain(zip_path)

        
        # Display each PDF resume on the webpage
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            if file_name.lower().endswith('.txt'):
                st.write(f"### {file_name}")
                with open(file_path, 'rb') as pdf_file:
                    pdf_data = pdf_file.read()
                    st.download_button(
                        label=f"Download {file_name}",
                        data=pdf_data,
                        file_name=file_name,
                        mime="application/pdf"
                    )
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{pdf_data.encode("base64")}" width="700" height="500"></iframe>',
                        unsafe_allow_html=True
                    )

        # Display the results
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
else:
    st.write('Please upload a zip file containing resume files.')

st.markdown("""
    ---
    Made with ❤️ by Samudraneel Sarkar [LinkedIn](https://www.linkedin.com/in/samudraneel-sarkar) | [GitHub](https://github.com/samudraneel05)
     &   Guransh Goyal [LinkedIn](https://www.linkedin.com/in/guransh-goyal) | [GitHub](https://github.com/GuranshGoyal)
""")

st.markdown("""
    © 2025 P-125. All rights reserved.
""")
