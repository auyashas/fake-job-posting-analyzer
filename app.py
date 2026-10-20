
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fraudulent Job Posting Detector",
    page_icon="🔎",
    layout="centered"
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "fraudulent_job_detector.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# Application title
st.title("🔎 Fraudulent Job Posting Detector")
st.write(
    "Analyze a job posting using our trained Machine Learning model "
    "to estimate whether it may be fraudulent."
)

st.info(
    "This tool provides a prediction, not a guarantee. "
    "Always verify suspicious job offers independently."
)

# Job posting details
st.header("Enter Job Posting Details")

with st.form("job_posting_form"):
    title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    company_profile = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    benefits = st.text_area("Benefits")

    st.subheader("Additional Information")

    location = st.text_input("Location", placeholder="e.g., Bengaluru, India")
    department = st.text_input("Department")
    salary_range = st.text_input("Salary Range", placeholder="e.g., 5-8 LPA")

    employment_type = st.selectbox(
        "Employment Type",
        ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
    )

    required_experience = st.selectbox(
        "Required Experience",
        [
            "", "Internship", "Entry level", "Associate",
            "Mid-Senior level", "Director", "Executive", "Other"
        ]
    )

    required_education = st.selectbox(
        "Required Education",
        [
            "", "High School or equivalent", "Vocational",
            "Some College Coursework Completed", "Associate Degree",
            "Bachelor's Degree", "Master's Degree", "Doctorate", "Other"
        ]
    )

    industry = st.text_input("Industry")
    function = st.text_input("Job Function")

    telecommuting = st.selectbox(
        "Is the job remote?",
        ["No", "Yes"]
    )

    has_company_logo = st.selectbox(
        "Does the posting include a company logo?",
        ["Yes", "No"]
    )

    has_questions = st.selectbox(
        "Does the application include screening questions?",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button(
        "Analyze Job Posting",
        type="primary"
    )

if submitted:
    # Validate required fields before prediction
    if not title.strip() or not description.strip():
        st.warning(
            "Please enter both the Job Title and Job Description "
            "before analyzing the posting."
        )
    else:
        # Combine text fields in the same order used during training
        combined_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits
        ])

        # Match the features used during model training
        input_data = pd.DataFrame([{
            "location": location or None,
            "department": department or None,
            "salary_range": salary_range or None,
            "telecommuting": 1 if telecommuting == "Yes" else 0,
            "has_company_logo": 1 if has_company_logo == "Yes" else 0,
            "has_questions": 1 if has_questions == "Yes" else 0,
            "employment_type": employment_type or None,
            "required_experience": required_experience or None,
            "required_education": required_education or None,
            "industry": industry or None,
            "function": function or None,
            "combined_text": combined_text
        }])

        try:
            prediction = int(model.predict(input_data)[0])

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Potentially Fraudulent Job Posting")
                st.write(
                    "The model classified this posting as potentially "
                    "fraudulent. Verify the employer and job details "
                    "before proceeding."
                )
            else:
                st.success("✅ Likely Legitimate Job Posting")
                st.write(
                    "The model classified this posting as legitimate. "
                    "This does not guarantee that the job offer is genuine."
                )

        except Exception as e:
            st.error(f"Unable to analyze this posting: {e}")

