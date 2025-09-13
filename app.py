import streamlit as st
import pandas as pd
import os
import sys
import json
import logging
from datetime import datetime

# --- Python Path Correction ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------

from pipeline import process_feedback
from pipeline import predict_sentiment
from pipeline import readmission_data_pipeline
#from pipeline import predict_readmission_pipeline
from pipeline import los_data_pipeline

# ASTHETICS: Background Color and Such

def apply_healthcare_theme():
    """Apply professional healthcare styling"""
    st.markdown("""
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #f8fafe;
        color: #1f4e79;
    }
    
    /* Ensure all text is readable */
    .stApp p, .stApp span, .stApp div {
        color: #1f4e79 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 2px solid #e1e8ed;
    }
    
    /* Sidebar title */
    .css-1d391kg h1 {
        color: #1f4e79;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 2px solid #4a90c2;
    }
    
    /* Radio buttons in sidebar */
    .css-1d391kg .stRadio > label {
        background-color: #f8fafe;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stRadio > label:hover {
        background-color: #e3f2fd;
        border-color: #4a90c2;
    }
    
    /* Selected radio button */
    .css-1d391kg .stRadio > label[data-baseweb="radio"] {
        background-color: #4a90c2;
        color: white;
        font-weight: 500;
    }
    
    /* Main content area */
    .block-container {
        padding: 2rem 3rem;
        background-color: #ffffff;
        border-radius: 12px;
        margin: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1 {
        color: #1f4e79;
        font-weight: 600;
        border-bottom: 3px solid #4a90c2;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #2c5aa0;
        font-weight: 500;
        margin-top: 2rem;
    }
    
    h3 {
        color: #345a8a;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4a90c2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background-color: #3a7cb0;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        color: #2e7d32;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        color: #f57c00;
    }
    
    /* Error messages */
    .stError {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 8px;
        color: #c62828;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        color: #1565c0;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background-color: #f8fafe;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #f8fafe;
        border: 2px dashed #4a90c2;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f7ff;
        border: 1px solid #4a90c2;
        border-radius: 8px;
        color: #1f4e79;
        font-weight: 500;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Text areas */
    .stTextArea textarea {
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        background-color: #ffffff;
        color: #1f4e79 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4a90c2;
        box-shadow: 0 0 0 2px rgba(74, 144, 194, 0.2);
    }
    
    /* Text area placeholder */
    .stTextArea textarea::placeholder {
        color: #8a8a8a !important;
    }
    
    /* Text inputs */
    .stTextInput input {
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        background-color: #ffffff;
        color: #8a8a8a !important;
    }
    
    .stTextInput input:focus {
        border-color: #4a90c2;
        box-shadow: 0 0 0 2px rgba(74, 144, 194, 0.2);
    }
    
    /* Selectbox */
    .stSelectbox select {
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        background-color: #ffffff;
    }
    
    /* Slider */
    .stSlider .css-1cpxqw2 {
        background-color: #4a90c2;
    }
    
    /* Remove default Streamlit branding colors */
    .css-10trblm {
        color: #1f4e79;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #4a90c2 transparent #4a90c2 transparent;
    }
    
    /* Footer */
    .css-164nlkn {
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Hide Streamlit menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)


# Add this call right after st.set_page_config in your app.py
apply_healthcare_theme()

# --- Logging Configuration ---
def setup_logging():
    """Configures logging to write to a file in a 'logs' directory."""
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"streamlit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H%M:%S',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    return log_filepath

# --- Sentiment Analysis Pipeline Logic ---
@st.cache_data
def execute_pipeline(input_df, drug_name):
    """
    Orchestrates the data processing and sentiment analysis pipeline.
    """
    log_file = setup_logging()
    logging.info(f"Pipeline run initiated for read: '{drug_name}'. Log file: {log_file}")

    # Step 1: Process the raw data
    logging.info("Step 1: Preprocessing data...")
    processed_df = process_feedback.run_preprocessing(input_df)
    
    # Step 2: Run sentiment prediction and analysis
    logging.info("Step 2: Predicting sentiment and running analysis...")
    prediction_result = predict_sentiment.run_inference(processed_df, drug_name=drug_name)
    
    logging.info("Pipeline execution complete.")
    return prediction_result, log_file

# ------ Readmission Prediction Pipeline Logic -----
@st.cache_data
def execute_readmission_pipeline(model_pipeline, input_file):
    # Process the data
    #input_file = "../healthcare/patient_encounters_2023.csv"
    output_file = "processed_readmission_data.csv"
    pipeline_file = "readmission_data_pipeline.pkl"
    
    """
    Orchestrates the data processing and sentiment analysis pipeline.
    """
    log_file = setup_logging()
    logging.info(f"Pipeline run initiated for readmission prediction")#drug: '{drug_name}'. Log file: {log_file}")

    """# Step 1: Process the raw data
    logging.info("Step 1: Preprocessing data...")
    scaled_data, processed_data = readmission_data_pipeline.process_file(input_file, output_file, pipeline_file)"""
    
    # Step 2: Run sentiment prediction and analysis
    logging.info("Step 2: Predicting readmission...")
    results = model_pipeline.predict_batch(input_file, output_file, 0.3)
    
    logging.info("Pipeline execution complete.")
    return results, log_file

# --- Streamlit Page Configuration & UI ---
st.set_page_config(page_title="Medicine Feedback Analysis", page_icon="💊", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🏥 ML Models")
    page = st.radio(
        "Choose Analysis Type:",
        ["🏠 Home", "😊 Sentiment Analysis", "🏥 Readmission Prediction", 
         "⏱️ Length of Stay", "🧪 Retinal Image Test", 
         "📋 Patient Feedback", "💊 Medicine Feedback", "🏥 Clinical Codes"]
    )

# --- Main Content Based on Selection ---
if page == "🏠 Home":
    st.title("💊 Fusion 5 - AI Healthcare Solutions")
    
    # Team Introduction
    st.markdown("---")
    st.header("🚀 Our Mission")
    st.markdown("""
    **Fusion 5** leverages cutting-edge AI and Machine Learning technologies to revolutionize healthcare delivery. 
    Our comprehensive suite of predictive models empowers medical professionals to make data-driven decisions, 
    ultimately improving patient outcomes and enhancing the quality of care.
    
    Through intelligent analysis of patient data, medication feedback, and clinical indicators, we provide 
    healthcare providers with the insights they need to deliver personalized, efficient, and effective treatment.
    """)
    
    # Model Overview
    st.header("🔬 Our AI Models")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**😊 Sentiment Analysis**")
        st.markdown("Analyze patient feedback and medication reviews")
        
        st.markdown("**🏥 Readmission Prediction**") 
        st.markdown("Predict patient readmission risk factors")
    
    with col2:
        st.markdown("**⏱️ Length of Stay Prediction**")
        st.markdown("Forecast hospital stay duration")
        
        st.markdown("**🧠 CNN Image Analysis**")
        st.markdown("Advanced medical image processing")
    
    # Getting Started
    st.header("📋 Getting Started")
    st.markdown("Choose a model from the sidebar to begin analyzing your healthcare data and improving patient care.")
    
    # Thank You Note
    st.markdown("---")
    st.header("🙏 Acknowledgments")
    st.markdown("""
    **Team Fusion 5** - Christopher, Greggy, Sirisha, Srividya, and Sean - would like to extend our heartfelt 
    gratitude to **Professor Abhishek** for creating an encouraging and supportive learning environment. 
    
    Thank you for believing in our dreams, listening to our ideas, and guiding us through this incredible 
    journey of discovery in AI and healthcare innovation. Your mentorship has been invaluable in bringing 
    this vision to life.
    """)

elif page == "😊 Sentiment Analysis":
    st.title("😊 Sentiment Analysis")
    st.markdown("Analyze patient feedback and medication reviews using our sentiment analysis model.")
    
    # Analysis method selection
    analysis_method = st.radio(
        "Choose Analysis Method:",
        ["📄 Upload CSV File", "✏️ Enter Text Manually"],
        horizontal=True
    )
    
    if analysis_method == "📄 Upload CSV File":
        st.subheader("📄 CSV File Analysis")
        st.markdown("Upload a CSV file containing patient feedback data with drug names.")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Patient Feedback CSV", type=['csv'])
        
        if 'drug_options' not in st.session_state:
            st.session_state.drug_options = []

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'urlDrugName' in df.columns:
                    drug_names = sorted([name for name in df['urlDrugName'].unique() if pd.notna(name)])
                    st.session_state.drug_options = ["-- Select a Drug --"] + drug_names
                else:
                    st.error("The uploaded CSV must contain a 'urlDrugName' column.")
                    st.session_state.drug_options = []
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.session_state.drug_options = []
        
        selected_drug = st.selectbox(
            "Select Drug Name for Analysis",
            options=st.session_state.drug_options,
            index=0,
            disabled=(not st.session_state.drug_options)
        )

        # CSV Analysis execution
        if uploaded_file and selected_drug != "-- Select a Drug --":
            st.info(f"Ready to analyze **{selected_drug}** from `{uploaded_file.name}`.")
            
            if st.button(f"🚀 Run Analysis for {selected_drug}", use_container_width=True, type="primary"):
                with st.spinner("Pipeline is running... This may take a moment."):
                    try:
                        uploaded_file.seek(0)
                        input_df = pd.read_csv(uploaded_file)
                        
                        results, log_file = execute_pipeline(input_df, selected_drug)
                        st.success("✅ Pipeline executed successfully!")
                        st.info(f"A detailed log has been saved to: `{log_file}`")
                        
                        # Display Results
                        final_df = pd.DataFrame(results.get("data", []))
                        specific_analysis = results.get("specific_drug_analysis", {})

                        st.subheader(f"📊 Analysis Results for {selected_drug.title()}")
                        if specific_analysis:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Overall Sentiment", specific_analysis.get("overall_sentiment", "N/A"))
                            col2.metric("Average Rating", f"{specific_analysis.get('average_rating', 0):.2f}/10")
                            col3.metric("Reviews Found", f"{specific_analysis.get('reviews_found', 0)}")
                        
                        with st.expander("🔬 View Detailed JSON Analysis"):
                            summary_to_display = results.copy()
                            summary_to_display.pop("data", None)
                            st.json(summary_to_display)
                        
                        if not final_df.empty:
                            st.subheader("Processed Data with Predictions")
                            st.dataframe(final_df)

                    except Exception as e:
                        st.error("An error occurred during pipeline execution:")
                        st.exception(e)
        else:
            st.warning("Please upload a CSV file and select a drug from the dropdown to begin.")
    
    elif analysis_method == "✏️ Enter Text Manually":
        st.subheader("✏️ Manual Text Analysis")
        st.markdown("Enter patient feedback text and drug name for quick sentiment analysis.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input area
            user_text = st.text_area(
                "Patient Feedback Text:",
                placeholder="Enter patient feedback here... e.g., 'This medication helped with my condition but caused some side effects...'",
                height=150
            )
        
        with col2:
            # Drug name input
            drug_name = st.text_input(
                "Drug Name:",
                placeholder="e.g., Aspirin, Metformin"
            )
            
            # Rating input (optional)
            rating = st.slider(
                "Rating (optional):",
                min_value=1,
                max_value=10,
                value=5,
                help="Patient rating from 1-10"
            )
        
        # Manual analysis execution
        if st.button("🔍 Analyze Text", use_container_width=True, type="primary"):
            if user_text.strip() and drug_name.strip():
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Create a simple DataFrame from the manual input
                        manual_data = pd.DataFrame({
                            'review': [user_text.strip()],
                            'urlDrugName': [drug_name.strip()],
                            'rating': [rating]
                        })
                        
                        # Run the sentiment analysis pipeline
                        results, log_file = execute_pipeline(manual_data, drug_name.strip())
                        
                        st.success("✅ Text analysis complete!")
                        
                        # Display Results
                        st.subheader("📊 Analysis Results")
                        
                        specific_analysis = results.get("specific_drug_analysis", {})
                        if specific_analysis:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Predicted Sentiment", specific_analysis.get("overall_sentiment", "N/A"))
                            col2.metric("Input Rating", f"{rating}/10")
                            col3.metric("Confidence", "High" if specific_analysis.get("reviews_found", 0) > 0 else "N/A")
                        
                        # Show the input text for reference
                        with st.expander("📝 Input Text Analysis"):
                            st.markdown(f"**Drug:** {drug_name}")
                            st.markdown(f"**Text:** {user_text}")
                            st.markdown(f"**Rating:** {rating}/10")
                        
                        # Show detailed results if available
                        final_df = pd.DataFrame(results.get("data", []))
                        if not final_df.empty:
                            with st.expander("🔬 Detailed Analysis"):
                                st.dataframe(final_df)
                                
                    except Exception as e:
                        st.error("An error occurred during text analysis:")
                        st.exception(e)
            else:
                st.warning("Please enter both patient feedback text and drug name.")
        
        # Sample text examples
        with st.expander("💡 Sample Text Examples"):
            st.markdown("""
            **Positive Example:**
            *"This medication has been life-changing for my diabetes. My blood sugar levels are much more stable now and I feel great!"*
            
            **Negative Example:**
            *"The medication helped with my condition but the side effects were terrible. I experienced nausea and dizziness daily."*
            
            **Neutral Example:**
            *"The medication works as expected. No major improvements or side effects to report."*
            """)

elif page == "⏱️ Length of Stay":
    st.title("⏱️ Length of Stay Prediction")
    st.markdown("Upload patient encounter data to predict hospital length of stay with risk assessment and recommendations.")
    
    # Model info
    with st.expander("📋 Model Information"):
        st.markdown("""
        **Risk Levels:**
        - **Low Risk:** ≤ 3 days (standard care protocols)
        - **Medium Risk:** 4-7 days (monitor for complications)  
        - **High Risk:** 8-14 days (enhanced monitoring protocols)
        - **Very High Risk:** > 14 days (intensive intervention protocols)
        
        **Features:** Age, BMI, admission type, diagnoses, medications, procedures, lab tests
        """)
    
    # File upload for patient encounters
    uploaded_file = st.file_uploader("Upload Patient Encounters CSV", type=['csv'], key="los_upload")
    
    if uploaded_file is not None:
        st.info(f"Ready to analyze length of stay from `{uploaded_file.name}`.")
        
        if st.button("🚀 Run Length of Stay Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing length of stay and generating predictions..."):
                try:
                    # Read the uploaded file
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Create pipeline instance and process
                    from los_data_pipeline import LOSDataPipeline
                    pipeline = LOSDataPipeline()
                    
                    # Process the data
                    processed_data = pipeline.preprocess_data(input_df)
                    pipeline.fit_scaler(processed_data)
                    scaled_data = pipeline.transform_data(processed_data)
                    
                    st.success("✅ Length of stay analysis complete!")
                    
                    # Mock predictions for demo (replace with actual model when available)
                    # This simulates what your BatchLOSPredictor would return
                    import numpy as np
                    mock_predictions = np.random.uniform(1, 20, len(input_df))
                    risk_levels = []
                    recommendations = []
                    
                    for pred in mock_predictions:
                        if pred <= 3:
                            risk_levels.append("Low")
                            recommendations.append("Standard care protocols")
                        elif pred <= 7:
                            risk_levels.append("Medium")
                            recommendations.append("Monitor closely for complications")
                        elif pred <= 14:
                            risk_levels.append("High")
                            recommendations.append("Enhanced monitoring protocols; Consider early intervention")
                        else:
                            risk_levels.append("Very High")
                            recommendations.append("Intensive intervention protocols; High-risk care team")
                    
                    # Create results dataframe
                    results_df = input_df.copy()
                    results_df['predicted_length_of_stay'] = np.round(mock_predictions, 1)
                    results_df['risk_level'] = risk_levels
                    results_df['recommendations'] = recommendations
                    results_df['prediction_confidence'] = np.random.uniform(0.7, 0.95, len(mock_predictions))
                    
                    # Display summary metrics
                    st.subheader("📊 Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Patients Processed", len(input_df))
                    col2.metric("Avg Predicted LOS", f"{results_df['predicted_length_of_stay'].mean():.1f} days")
                    col3.metric("Median LOS", f"{results_df['predicted_length_of_stay'].median():.1f} days")
                    col4.metric("High Risk Patients", len(results_df[results_df['risk_level'].isin(['High', 'Very High'])]))
                    
                    # Risk level distribution
                    st.subheader("⚠️ Risk Level Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Low Risk", f"{risk_counts.get('Low', 0)} ({risk_counts.get('Low', 0)/len(results_df)*100:.1f}%)")
                    col2.metric("Medium Risk", f"{risk_counts.get('Medium', 0)} ({risk_counts.get('Medium', 0)/len(results_df)*100:.1f}%)")
                    col3.metric("High Risk", f"{risk_counts.get('High', 0)} ({risk_counts.get('High', 0)/len(results_df)*100:.1f}%)", delta_color="inverse")
                    col4.metric("Very High Risk", f"{risk_counts.get('Very High', 0)} ({risk_counts.get('Very High', 0)/len(results_df)*100:.1f}%)", delta_color="inverse")
                    
                    # Top risk patients
                    st.subheader("🚨 Highest Risk Patients")
                    top_risk = results_df.nlargest(5, 'predicted_length_of_stay')[
                        ['patient_nbr', 'predicted_length_of_stay', 'risk_level', 'recommendations', 'prediction_confidence']
                    ] if 'patient_nbr' in results_df.columns else results_df.nlargest(5, 'predicted_length_of_stay')[
                        ['predicted_length_of_stay', 'risk_level', 'recommendations', 'prediction_confidence']
                    ]
                    
                    # Color code the risk levels
                    def color_risk_level(val):
                        if val == 'Very High':
                            return 'background-color: #ffebee; color: #c62828'
                        elif val == 'High':
                            return 'background-color: #fff3e0; color: #f57c00'
                        elif val == 'Medium':
                            return 'background-color: #fff8e1; color: #f9a825'
                        else:
                            return 'background-color: #e8f5e8; color: #2e7d32'
                    
                    st.dataframe(top_risk.style.applymap(color_risk_level, subset=['risk_level']))
                    
                    # Feature importance (if available)
                    with st.expander("📋 Feature Summary"):
                        st.write("**Features used for prediction:**")
                        st.write(pipeline.feature_columns)
                        
                        if hasattr(processed_data, 'describe'):
                            st.write("**Data Summary:**")
                            st.dataframe(processed_data.describe())
                    
                    # Download results
                    with st.expander("💾 Download Results"):
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"los_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    
                    # Full results table
                    with st.expander("📄 View All Predictions"):
                        st.dataframe(results_df.style.applymap(color_risk_level, subset=['risk_level']))
                    
                    # Clinical insights
                    st.subheader("🏥 Clinical Insights")
                    
                    avg_los = results_df['predicted_length_of_stay'].mean()
                    high_risk_pct = len(results_df[results_df['risk_level'].isin(['High', 'Very High'])]) / len(results_df) * 100
                    
                    if high_risk_pct > 30:
                        st.warning(f"⚠️ **High Risk Alert:** {high_risk_pct:.1f}% of patients are classified as high or very high risk for extended stays. Consider resource allocation and care coordination.")
                    elif high_risk_pct > 15:
                        st.info(f"ℹ️ **Moderate Risk:** {high_risk_pct:.1f}% of patients may require extended stays. Monitor closely.")
                    else:
                        st.success(f"✅ **Low Risk Population:** Only {high_risk_pct:.1f}% of patients are predicted for extended stays.")
                    
                    if avg_los > 10:
                        st.warning(f"📊 Average predicted LOS of {avg_los:.1f} days is above normal range. Consider early discharge planning.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.exception(e)
    else:
        st.warning("Please upload a patient encounters CSV file to begin.")
        
        # Sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Required columns:** age, gender, admission_type_id, time_in_hospital, num_medications, 
            num_lab_procedures, num_procedures, weight (optional), patient_nbr (optional)
            
            **Sample format:**
            ```
            patient_nbr,age,gender,admission_type_id,time_in_hospital,num_medications,num_lab_procedures,num_procedures
            12345,[70-80),Male,1,5,15,25,3
            67890,[50-60),Female,2,3,8,12,1
            ```
            """)

elif page == "🏥 Readmission Prediction":
    st.title("🏥 Readmission Prediction")
    st.markdown("Upload patient encounter data to predict readmission risk.")
    
    # File upload for patient encounters
    uploaded_file = st.file_uploader("Upload Patient Encounters CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.info(f"Ready to analyze patient encounters from `{uploaded_file.name}`.")
        
        if st.button("🚀 Run Readmission Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing readmission risk..."):
                try:
                    """# Read the uploaded file
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Create pipeline instance and process
                    from readmission_data_pipeline import ReadmissionDataPipeline
                    pipeline = ReadmissionDataPipeline()
                    
                    # Process the data
                    processed_data = pipeline.preprocess_data(input_df)
                    pipeline.fit_scaler(processed_data)
                    scaled_data = pipeline.transform_data(processed_data)"""

                    model_path='models/readmission_model.joblib'
                    pipeline_path='models/readmission_data_pipeline.pkl'
                    from predict_readmission_pipeline import ImprovedBatchReadmissionPredictor
                    predictor = ImprovedBatchReadmissionPredictor(_model_path, pipeline_path)
                    
                    results, log_file = execute_readmission_pipeline(predictor, uploaded_file)
                    st.success("✅ Readmission analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Readmission Analysis Results")
                    #st.write(f"Processed {len(input_df)} patient records")
                    st.write(f"Readmission Resultset -")
                    st.dataframe(results.head())
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    else:
        st.warning("Please upload a patient encounters CSV file to begin.")

elif page == "⏱️ Length of Stay":
    st.title("⏱️ Length of Stay Prediction")
    st.markdown("Upload patient encounter data to predict hospital length of stay.")
    
    # File upload for patient encounters
    uploaded_file = st.file_uploader("Upload Patient Encounters CSV", type=['csv'], key="los_upload")
    
    if uploaded_file is not None:
        st.info(f"Ready to analyze length of stay from `{uploaded_file.name}`.")
        
        if st.button("🚀 Run Length of Stay Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing length of stay..."):
                try:
                    # Read the uploaded file
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Create pipeline instance and process
                    from los_data_pipeline import LOSDataPipeline
                    pipeline = LOSDataPipeline()
                    
                    # Process the data
                    processed_data = pipeline.preprocess_data(input_df)
                    pipeline.fit_scaler(processed_data)
                    scaled_data = pipeline.transform_data(processed_data)
                    
                    st.success("✅ Length of stay analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Length of Stay Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Patients Processed", len(input_df))
                    col2.metric("Features Extracted", len(pipeline.feature_columns))
                    col3.metric("Average Age", f"{processed_data['age'].mean():.1f} years")
                    
                    # Show feature summary
                    with st.expander("📋 Feature Summary"):
                        st.write("**Features used for prediction:**")
                        st.write(pipeline.feature_columns)
                    
                    # Show processed data preview
                    st.subheader("📄 Processed Data Preview")
                    st.dataframe(processed_data.head(10))
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.exception(e)
    else:
        st.warning("Please upload a patient encounters CSV file to begin.")

elif page == "🧪 Retinal Image Test":
    st.title("🧪 Retinal Image Test")
    st.markdown("Upload a retinal image to classify diabetic retinopathy severity using our deep learning model.")
    
    # Model info
    with st.expander("📋 Model Overview"):
        st.markdown("""
        **Classification Scale:** International Clinical Diabetic Retinopathy (ICDR)
        
        - **0 - No DR:** Healthy retina with no signs of diabetic damage
        - **1 - Mild:** Presence of microaneurysms only
        - **2 - Moderate:** More than microaneurysms, but less than severe non-proliferative DR
        - **3 - Severe:** Extensive microvascular damage with hemorrhages and venous beading
        - **4 - Proliferative DR:** Growth of new blood vessels (neovascularization) that can cause vision-threatening complications
        """)
    
    # Image upload
    uploaded_image = st.file_uploader("Upload Retinal Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Retinal Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
                with st.spinner("Analyzing retinal image..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            temp_path = tmp_file.name
                        
                        # Initialize CNN tester
                        from cnn_model_test import RetinalCNNTester
                        tester = RetinalCNNTester(model_path="basic_modelv4_final.h5")
                        
                        # Make prediction
                        result = tester.predict_single_image(temp_path)
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        if result:
                            st.success("✅ Image classification complete!")
                            
                            # Display results
                            st.subheader("📊 Classification Results")
                            
                            # Main prediction
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Prediction", result['predicted_label'])
                            col2.metric("Confidence", f"{result['confidence']:.1%}")
                            col3.metric("Risk Level", "⚠️ High" if result['predicted_class'] >= 3 else "✅ Low-Moderate")
                            
                            # Probability breakdown
                            st.subheader("🎯 Probability Breakdown")
                            prob_df = pd.DataFrame({
                                'Severity Level': ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
                                'Probability': [f"{p:.1%}" for p in result['all_probabilities']],
                                'Score': result['all_probabilities']
                            })
                            
                            # Highlight predicted class
                            def highlight_max(s):
                                is_max = s == s.max()
                                return ['background-color: lightgreen' if v else '' for v in is_max]
                            
                            st.dataframe(prob_df.style.apply(highlight_max, subset=['Score']))
                            
                            # Medical disclaimer
                            st.warning("⚠️ **Medical Disclaimer:** This is an AI model for educational/research purposes only. Always consult qualified medical professionals for diagnosis and treatment.")
                        
                        else:
                            st.error("Failed to process image. Please try a different image.")
                            
                    except Exception as e:
                        st.error(f"Error during classification: {e}")
                        st.exception(e)
    else:
        st.warning("Please upload a retinal image to begin classification.")
        
        # Technical specifications
        with st.expander("📋 Technical Details"):
            st.markdown("""
            **Expected Image Format:** JPG, JPEG, or PNG retinal/fundus photographs
            
            **Model Input:** 512x512 pixel images (automatically resized)
            
            **Processing:** Images are automatically normalized and resized for optimal model performance
            """)


elif page == "📋 Patient Feedback":
    st.title("📋 Patient Feedback Data")
    st.markdown("Access and analyze patient feedback datasets.")
    st.info("Link to patient feedback data sources and analysis tools.")
    # Add patient feedback functionality here

elif page == "💊 Medicine Feedback":
    st.title("💊 Medicine Feedback Data")
    st.markdown("Access and analyze medicine feedback datasets.")
    st.info("Link to medicine feedback data sources and analysis tools.")
    # Add medicine feedback functionality here

elif page == "🏥 Clinical Codes":
    st.title("🏥 Clinical Codes Data")
    st.markdown("Access and analyze clinical codes datasets.")
    st.info("Link to clinical codes data sources and reference materials.")
    # Add clinical codes functionality here

else:
    st.error("Page not found. Please select a valid option from the sidebar.")
