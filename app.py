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
from pipeline import los_data_pipeline
from predict_readmission_pipeline import ImprovedBatchReadmissionPredictor
from cnn_model_test import RetinalCNNTester
from enhanced_retinal_cnn import EnhancedRetinalCNNv2

# ASTHETICS: Background Color and Such

# Add this CSS styling function right after your imports in app.py

# Add this CSS styling function right after your imports in app.py

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
    
    /* JSON display - more aggressive targeting */
    .stJson, .stJson div, .stJson pre, .stJson code {
        background-color: #ffffff !important;
        color: #1f4e79 !important;
    }
    
    /* Target all JSON-related elements */
    [data-testid="stJson"], [data-testid="stJson"] * {
        background-color: #ffffff !important;
        color: #1f4e79 !important;
    }
    
    /* Override any dark theme JSON styles */
    .css-1629p8f, .css-1629p8f pre, .css-1629p8f code {
        background-color: #ffffff !important;
        color: #1f4e79 !important;
    }
    
    /* JSON container */
    .element-container .stJson {
        background: #ffffff !important;
        border: 1px solid #e1e8ed !important;
        border-radius: 8px !important;
    }
    
    /* JSON syntax highlighting override */
    .stJson .token.string {
        color: #2e7d32 !important;
    }
    
    .stJson .token.number {
        color: #1976d2 !important;
    }
    
    .stJson .token.boolean {
        color: #7b1fa2 !important;
    }
    
    .stJson .token.null {
        color: #f57c00 !important;
    }
    
    .stJson .token.property {
        color: #1f4e79 !important;
        font-weight: 500;
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
def execute_readmission_pipeline(input_file):
    """
    Orchestrates the data processing and risk analysis pipeline.
    """
    log_file = setup_logging()
    logging.info(f"Pipeline run initiated for readmission prediction. Log file: {log_file}")

    # Step 1: Process the raw data and provide predictions
    logging.info("Step 1: Preprocessing data and computing predictions...")
    pipeline = ImprovedBatchReadmissionPredictor()
    results = pipeline.predict_batch(uploaded_file, None, 0.5)

    # Create result subset for visualization
    results_subset = results[['encounter_id', 'patient_nbr', 'readmitted', 'predicted_readmission', 'readmission_probability', 'risk_level', 'recommendations', 'prediction_confidence']]

    logging.info("Pipeline execution complete.")
    return results_subset, log_file

# --- Streamlit Page Configuration & UI ---
st.set_page_config(page_title="Medicine Feedback Analysis", page_icon="💊", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🏥 ML Models")
    page = st.radio(
        "Choose Analysis Type:",
        ["🏠 Home", "😊 Sentiment Analysis", "🏥 Readmission Prediction", 
         "⏱️ Length of Stay", "🧪 Retinal Image Test", "🗂️ Retinopathy Bulk Testing",
         "🏥 Clinical Codes", "📞 Fusion5 Contact"]
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
###____________________________________________________LOS______________________________________________________________________________________________
elif page == "⏱️ Length of Stay":
    st.title("⏱️ Length of Stay Prediction")
    st.markdown("Upload patient encounter data to predict hospital length of stay with risk assessment and clinical recommendations.")
    
    # Model info
    with st.expander("📋 Model Information"):
        st.markdown("""
        **Risk Assessment Levels:**
        - **Low Risk:** ≤ 3 days - Standard care protocols
        - **Medium Risk:** 4-7 days - Monitor closely for complications  
        - **High Risk:** 8-14 days - Enhanced monitoring protocols
        - **Very High Risk:** > 14 days - Consider early intervention protocols
        
        **Clinical Recommendations:**
        - Enhanced geriatric care for patients > 65 years
        - Nutritional counseling for BMI > 30
        - Early intervention for predicted stays > 10 days
        
        **Model Features:** Age, gender, admission type, diagnoses, medications, procedures, lab tests, BMI
        """)
    
    # File upload for patient encounters
    uploaded_file = st.file_uploader("Upload Patient Encounters CSV", type=['csv'], key="los_upload")
    
    if uploaded_file is not None:
        st.info(f"Ready to analyze length of stay from `{uploaded_file.name}`.")
        
        if st.button("🚀 Run Length of Stay Analysis", use_container_width=True, type="primary"):
            with st.spinner("Loading model and analyzing patient data..."):
                try:
                    # Save uploaded file temporarily for the predictor
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp_file:
                        uploaded_file.seek(0)
                        tmp_file.write(uploaded_file.getvalue().decode())
                        temp_csv_path = tmp_file.name
                    
                    # Check if model files exist
                    model_path = 'models/los_model.pkl'
                    pipeline_path = 'models/los_data_pipeline.pkl'
                    
                    if os.path.exists(model_path) and os.path.exists(pipeline_path):
                        # Use actual trained model
                        st.info("Loading trained LOS model...")
                        
                        # First check if the CSV has required columns
                        test_df = pd.read_csv(uploaded_file)
                        required_cols = ['age', 'gender', 'admission_type_id', 'time_in_hospital', 
                                       'num_medications', 'num_lab_procedures', 'num_procedures']
                        missing_cols = [col for col in required_cols if col not in test_df.columns]
                        
                        if missing_cols:
                            st.error(f"Missing required columns: {missing_cols}")
                            st.info("Please ensure your CSV contains the required columns listed in the expected format section below.")
                        else:
                            # Import your BatchLOSPredictor
                            from predict_lengthofstay_pipeline import BatchLOSPredictor
                            
                            # Initialize predictor with your trained models
                            predictor = BatchLOSPredictor(model_path, pipeline_path)
                            
                            # Make predictions
                            results_df = predictor.predict_batch(temp_csv_path)
                            
                            st.success("✅ Length of stay analysis complete using trained model!")
                        
                    else:
                        # Fallback to pipeline-only processing (no trained model available)
                        st.warning("Trained model not found. Using data processing pipeline only.")
                        
                        # Read the uploaded file
                        input_df = pd.read_csv(uploaded_file)
                        
                        # Create pipeline instance and process
                        from predict_lengthofstay_pipeline import BatchLOSPredictor
                        # Use just the pipeline functionality
                        from pipeline import los_data_pipeline
                        pipeline = los_data_pipeline.LOSDataPipeline()
                        
                        # Process the data
                        processed_data = pipeline.preprocess_data(input_df)
                        pipeline.fit_scaler(processed_data)
                        scaled_data = pipeline.transform_data(processed_data)
                        
                        # Generate mock predictions for demo
                        import numpy as np
                        mock_predictions = np.random.uniform(1, 20, len(input_df))
                        risk_levels = []
                        recommendations = []
                        
                        for i, pred in enumerate(mock_predictions):
                            # Use the same logic as your actual predictor
                            if pred <= 3:
                                risk_level = "Low"
                                rec = "Standard care protocols"
                            elif pred <= 7:
                                risk_level = "Medium"
                                rec = "Monitor closely for complications"
                            elif pred <= 14:
                                risk_level = "High"
                                rec = "Enhanced monitoring protocols"
                            else:
                                risk_level = "Very High"
                                rec = "Consider early intervention protocols"
                            
                            # Add age-based recommendations
                            if i < len(processed_data) and processed_data.iloc[i].get('age', 0) > 65:
                                rec += "; Enhanced geriatric care protocols"
                            
                            risk_levels.append(risk_level)
                            recommendations.append(rec)
                        
                        # Create results dataframe
                        results_df = input_df.copy()
                        results_df['predicted_length_of_stay'] = np.round(mock_predictions, 1)
                        results_df['risk_level'] = risk_levels
                        results_df['recommendations'] = recommendations
                        results_df['prediction_confidence'] = np.random.uniform(0.7, 0.95, len(mock_predictions))
                        
                        st.success("✅ Data processing complete! (Demo mode - install trained model for real predictions)")
                    
                    # Clean up temp file
                    os.unlink(temp_csv_path)
                    
                    # Display comprehensive results
                    st.subheader("📊 Prediction Summary")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_patients = len(results_df)
                    avg_los = results_df['predicted_length_of_stay'].mean()
                    median_los = results_df['predicted_length_of_stay'].median()
                    high_risk_count = len(results_df[results_df['risk_level'].isin(['High', 'Very High'])])
                    
                    col1.metric("Patients Analyzed", total_patients)
                    col2.metric("Average Predicted LOS", f"{avg_los:.1f} days")
                    col3.metric("Median Predicted LOS", f"{median_los:.1f} days")
                    col4.metric("High Risk Patients", high_risk_count, delta=f"{high_risk_count/total_patients*100:.1f}%")
                    
                    # Risk level distribution
                    st.subheader("⚠️ Risk Level Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    
                    risk_cols = st.columns(4)
                    risk_levels = ['Low', 'Medium', 'High', 'Very High']
                    risk_colors = ['normal', 'normal', 'inverse', 'inverse']
                    
                    for i, (level, color) in enumerate(zip(risk_levels, risk_colors)):
                        count = risk_counts.get(level, 0)
                        percentage = count/total_patients*100 if total_patients > 0 else 0
                        risk_cols[i].metric(
                            f"{level} Risk", 
                            f"{count} ({percentage:.1f}%)", 
                            delta_color=color
                        )
                    
                    # Clinical insights and alerts
                    st.subheader("🏥 Clinical Insights & Recommendations")
                    
                    high_risk_pct = high_risk_count / total_patients * 100 if total_patients > 0 else 0
                    
                    if high_risk_pct > 30:
                        st.error(f"🚨 **Critical Alert:** {high_risk_pct:.1f}% of patients are high/very high risk for extended stays. Immediate resource allocation and capacity planning required.")
                    elif high_risk_pct > 15:
                        st.warning(f"⚠️ **Elevated Risk:** {high_risk_pct:.1f}% of patients may require extended stays. Enhanced monitoring and care coordination recommended.")
                    else:
                        st.success(f"✅ **Manageable Risk:** {high_risk_pct:.1f}% of patients predicted for extended stays. Standard protocols sufficient.")
                    
                    if avg_los > 10:
                        st.warning(f"📊 **Extended Stay Alert:** Average predicted LOS of {avg_los:.1f} days exceeds normal range. Consider early discharge planning and case management.")
                    elif avg_los < 3:
                        st.info(f"📊 **Short Stay Pattern:** Average predicted LOS of {avg_los:.1f} days indicates efficient care delivery.")
                    
                    # Highest risk patients requiring immediate attention
                    st.subheader("🚨 Priority Patients (Highest Risk)")
                    
                    # Select columns that are likely to exist
                    display_cols = ['predicted_length_of_stay', 'risk_level', 'recommendations']
                    if 'patient_nbr' in results_df.columns:
                        display_cols.insert(0, 'patient_nbr')
                    if 'age' in results_df.columns:
                        display_cols.append('age')
                    if 'gender' in results_df.columns:
                        display_cols.append('gender')
                    
                    top_risk = results_df.nlargest(5, 'predicted_length_of_stay')[display_cols]
                    
                    # Color code the risk levels
                    def color_risk_level(val):
                        if val == 'Very High':
                            return 'background-color: #ffebee; color: #c62828; font-weight: bold'
                        elif val == 'High':
                            return 'background-color: #fff3e0; color: #f57c00; font-weight: bold'
                        elif val == 'Medium':
                            return 'background-color: #fff8e1; color: #f9a825'
                        else:
                            return 'background-color: #e8f5e8; color: #2e7d32'
                    
                    st.dataframe(
                        top_risk.style.applymap(color_risk_level, subset=['risk_level']),
                        use_container_width=True
                    )
                    
                    # Download and export options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download full results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv,
                            file_name=f"los_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    
                    with col2:
                        # Download high-risk patients only
                        high_risk_df = results_df[results_df['risk_level'].isin(['High', 'Very High'])]
                        if not high_risk_df.empty:
                            high_risk_csv = high_risk_df.to_csv(index=False)
                            st.download_button(
                                label="⚠️ Download High-Risk Patients (CSV)",
                                data=high_risk_csv,
                                file_name=f"high_risk_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                    
                    # Detailed results in expandable sections
                    with st.expander("📄 View All Predictions"):
                        st.dataframe(
                            results_df.style.applymap(color_risk_level, subset=['risk_level']),
                            use_container_width=True
                        )
                    
                    with st.expander("📊 Statistical Summary"):
                        st.write("**Length of Stay Statistics:**")
                        los_stats = results_df['predicted_length_of_stay'].describe()
                        st.dataframe(los_stats.to_frame().T)
                        
                        st.write("**Risk Level Breakdown:**")
                        risk_summary = results_df['risk_level'].value_counts().to_frame()
                        risk_summary.columns = ['Count']
                        risk_summary['Percentage'] = (risk_summary['Count'] / len(results_df) * 100).round(1)
                        st.dataframe(risk_summary)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.exception(e)
                    
    else:
        st.warning("Please upload a patient encounters CSV file to begin analysis.")
        
        # Expected data format help
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Required Columns:**
            - `age` - Patient age (can be ranges like [70-80))
            - `gender` - Male/Female/Unknown
            - `admission_type_id` - Admission type identifier
            - `time_in_hospital` - Current/historical length of stay
            - `num_medications` - Number of medications
            - `num_lab_procedures` - Number of lab procedures
            - `num_procedures` - Number of procedures
            - `weight` - Patient weight (optional, can be ranges)
            
            **Optional Columns:**
            - `patient_nbr` - Patient identifier
            - `race` - Patient race/ethnicity
            - Various diagnosis and medication columns
            
            **Sample Format:**
            ```csv
            patient_nbr,age,gender,admission_type_id,time_in_hospital,num_medications,num_lab_procedures,num_procedures,weight
            12345,[70-80),Male,1,5,15,25,3,[175-200)
            67890,[50-60),Female,2,3,8,12,1,[125-150)
            ```
            """)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
#-------------------------------------------------------------------------Readmisssion--------------------------------------------------------------------------
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
                    results, log_file = execute_readmission_pipeline(uploaded_file)
                    st.success("✅ Readmission analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Readmission Analysis Results")
                    st.write(f"Processed {len(results)} patient records")
                    st.write(f"Readmission Predictions")
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
#---------------------------------------------------------------------Retinal Image Test-----------------------------------------------------------------------------------------
elif page == "🧪 Retinal Image Test":
    st.title("🧪 Retinal Image Test")
    st.markdown("Upload a retinal image to classify diabetic retinopathy severity using our deep learning model.")
    
    # Model selection
    model_choice = st.radio(
        "🤖 Choose Model:",
        ["Basic CNN Model", "Enhanced CNN Model"],
        horizontal=True,
        help="Basic: Faster inference, Enhanced: More advanced architecture"
    )
    
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
                with st.spinner(f"Analyzing retinal image with {model_choice}..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            temp_path = tmp_file.name
                        
                        # Choose model based on selection
                        if model_choice == "Basic CNN Model":
                            # Basic model
                            from cnn_model_test import RetinalCNNTester
                            tester = RetinalCNNTester(model_path="basic_modelv4_final.h5")
                            result = tester.predict_single_image(temp_path)
                            
                        else:
                            # Enhanced model - with error handling
                            try:
                                from enhanced_retinal_cnn import EnhancedRetinalCNNv2
                                enhanced_cnn = EnhancedRetinalCNNv2()
                                
                                # Load model (adjust path as needed)
                                import tensorflow as tf
                                enhanced_cnn.model = tf.keras.models.load_model("enhanced_modelv4_final.h5")
                                
                                # Make prediction using enhanced model
                                result = enhanced_cnn.predict_single_image(temp_path)
                                
                            except ImportError:
                                st.error("Enhanced model not available. Using Basic model instead.")
                                from cnn_model_test import RetinalCNNTester
                                tester = RetinalCNNTester(model_path="basic_modelv4_final.h5")
                                result = tester.predict_single_image(temp_path)
                                
                            except Exception as e:
                                st.error(f"Enhanced model failed: {e}. Using Basic model instead.")
                                from cnn_model_test import RetinalCNNTester
                                tester = RetinalCNNTester(model_path="basic_modelv4_final.h5")
                                result = tester.predict_single_image(temp_path)
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        if result:
                            st.success("✅ Image classification complete!")
                            
                            # Show which model was used
                            st.info(f"🤖 **Model Used:** {model_choice}")
                            
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
                        st.error(f"Error during classification with {model_choice}: {e}")
                        st.exception(e)
    else:
        st.warning("Please upload a retinal image to begin classification.")
        
        # Technical specifications
        with st.expander("📋 Technical Details"):
            st.markdown("""
            **Expected Image Format:** JPG, JPEG, or PNG retinal/fundus photographs
            
            **Model Input:** 512x512 pixel images (automatically resized)
            
            **Processing:** Images are automatically normalized and resized for optimal model performance
            
            **Basic Model:** Lightweight CNN for fast inference
            
            **Enhanced Model:** Advanced architecture with attention mechanisms and progressive training
            """)
#-------------------------------------------------Bulk Tes---------------------------------------------------------------------------------------------------------------------------------------------
elif page == "🗂️ Retinopathy Bulk Testing":
    st.title("🗂️ Retinopathy Bulk Testing")
    st.markdown("Run **bulk predictions** (multiple images/CSV) in a separate Streamlit app.")

    _bulk_url = st.secrets.get("bulk_app_url") or os.getenv("BULK_APP_URL")
    if _bulk_url:
        try:
            st.page_link(_bulk_url, label="Open Retinopathy Bulk Testing app", icon="🗂️")
        except Exception:
            st.link_button("Open Retinopathy Bulk Testing app", _bulk_url, type="primary")
    else:
        st.warning("No bulk app URL configured. Please set `bulk_app_url` in `.streamlit/secrets.toml` "
                   "or `BULK_APP_URL` as an environment variable.")


#---------------------------------------------Codes--------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "🏥 Clinical Codes":
    st.title("🏥 Clinical Codes Reference")
    st.markdown("Medical coding reference for healthcare data analysis and interpretation.")
    
    # Create tabs for different code types
    tab1, tab2, tab3 = st.tabs(["🚑 Admission Types", "🏠 Discharge Dispositions", "📍 Admission Sources"])
    
    with tab1:
        st.subheader("Admission Type Codes")
        st.markdown("Classification of patient admission types and urgency levels.")
        
        admission_data = {
            "Code": [1, 2, 3, 4, 5, 6, 7, 8],
            "Description": [
                "Emergency",
                "Urgent", 
                "Elective",
                "Newborn",
                "Not Available",
                "NULL",
                "Trauma Center",
                "Not Mapped"
            ],
            "Category": [
                "Urgent Care",
                "Urgent Care",
                "Scheduled",
                "Birth",
                "Unknown",
                "Unknown", 
                "Critical Care",
                "Unknown"
            ]
        }
        
        admission_df = pd.DataFrame(admission_data)
        
        # Color code by category
        def color_admission_type(val):
            if val == "Urgent Care":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "Critical Care":
                return 'background-color: #e3f2fd; color: #1565c0'
            elif val == "Scheduled":
                return 'background-color: #e8f5e8; color: #2e7d32'
            elif val == "Birth":
                return 'background-color: #fff3e0; color: #f57c00'
            else:
                return 'background-color: #f5f5f5; color: #666'
        
        st.dataframe(
            admission_df.style.applymap(color_admission_type, subset=['Category']),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Codes", len(admission_df))
        col2.metric("Urgent/Critical", len(admission_df[admission_df['Category'].isin(['Urgent Care', 'Critical Care'])]))
        col3.metric("Scheduled", len(admission_df[admission_df['Category'] == 'Scheduled']))
    
    with tab2:
        st.subheader("Discharge Disposition Codes")
        st.markdown("Patient discharge destinations and care transitions.")
        
        discharge_data = {
            "Code": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            "Description": [
                "Discharged to home",
                "Discharged/transferred to another short term hospital", 
                "Discharged/transferred to SNF",
                "Discharged/transferred to ICF",
                "Discharged/transferred to another type of inpatient care institution",
                "Discharged/transferred to home with home health service",
                "Left AMA",
                "Discharged/transferred to home under care of Home IV provider",
                "Admitted as an inpatient to this hospital",
                "Neonate discharged to another hospital for neonatal aftercare",
                "Expired",
                "Still patient or expected to return for outpatient services",
                "Hospice / home",
                "Hospice / medical facility",
                "Discharged/transferred within this institution to Medicare approved swing bed",
                "Discharged/transferred/referred another institution for outpatient services",
                "Discharged/transferred/referred to this institution for outpatient services",
                "NULL",
                "Expired at home. Medicaid only, hospice",
                "Expired in a medical facility. Medicaid only, hospice",
                "Expired, place unknown. Medicaid only, hospice",
                "Discharged/transferred to another rehab fac including rehab units of a hospital",
                "Discharged/transferred to a long term care hospital",
                "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
                "Not Mapped",
                "Unknown/Invalid",
                "Discharged/transferred to a federal health care facility",
                "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
                "Discharged/transferred to a Critical Access Hospital (CAH)",
                "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere"
            ],
            "Category": [
                "Home", "Transfer", "Long-term Care", "Long-term Care", "Transfer", 
                "Home Care", "AMA", "Home Care", "Readmission", "Transfer",
                "Expired", "Outpatient", "Hospice", "Hospice", "Transfer",
                "Outpatient", "Outpatient", "Unknown", "Expired", "Expired", 
                "Expired", "Rehabilitation", "Long-term Care", "Long-term Care", "Unknown",
                "Unknown", "Transfer", "Psychiatric", "Transfer", "Transfer"
            ]
        }
        
        discharge_df = pd.DataFrame(discharge_data)
        
        # Color code by category
        def color_discharge_type(val):
            if val == "Home":
                return 'background-color: #e8f5e8; color: #2e7d32'
            elif val == "Home Care":
                return 'background-color: #f1f8e9; color: #388e3c'
            elif val == "Transfer":
                return 'background-color: #e3f2fd; color: #1565c0'
            elif val == "Long-term Care":
                return 'background-color: #fff3e0; color: #f57c00'
            elif val == "Expired":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "Hospice":
                return 'background-color: #fce4ec; color: #ad1457'
            elif val == "AMA":
                return 'background-color: #fff8e1; color: #f9a825'
            else:
                return 'background-color: #f5f5f5; color: #666'
        
        # Add search functionality
        search_term = st.text_input("Search discharge codes:", placeholder="Enter search term...")
        
        if search_term:
            filtered_df = discharge_df[discharge_df['Description'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = discharge_df
        
        st.dataframe(
            filtered_df.style.applymap(color_discharge_type, subset=['Category']),
            use_container_width=True,
            hide_index=True
        )
        
        # Category breakdown
        st.subheader("Discharge Category Summary")
        category_counts = discharge_df['Category'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            for i, (category, count) in enumerate(category_counts.head(4).items()):
                st.metric(category, count)
        with col2:
            for i, (category, count) in enumerate(category_counts.tail(len(category_counts)-4).items()):
                st.metric(category, count)
    
    with tab3:
        st.subheader("Admission Source Codes") 
        st.markdown("Origin points and referral sources for patient admissions.")
        
        admission_source_data = {
            "Code": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            "Description": [
                "Physician Referral",
                "Clinic Referral",
                "HMO Referral", 
                "Transfer from a hospital",
                "Transfer from a Skilled Nursing Facility (SNF)",
                "Transfer from another health care facility",
                "Emergency Room",
                "Court/Law Enforcement",
                "Not Available",
                "Transfer from critical access hospital",
                "Normal Delivery",
                "Premature Delivery",
                "Sick Baby",
                "Extramural Birth",
                "Not Available",
                "NULL",
                "Transfer From Another Home Health Agency",
                "Readmission to Same Home Health Agency",
                "Not Mapped",
                "Unknown/Invalid",
                "Transfer from hospital inpt/same fac reslt in a sep claim",
                "Born inside this hospital", 
                "Born outside this hospital",
                "Transfer from Ambulatory Surgery Center",
                "Transfer from Hospice"
            ],
            "Category": [
                "Referral", "Referral", "Referral", "Transfer", "Transfer",
                "Transfer", "Emergency", "Legal", "Unknown", "Transfer",
                "Birth", "Birth", "Birth", "Birth", "Unknown",
                "Unknown", "Transfer", "Readmission", "Unknown", "Unknown",
                "Transfer", "Birth", "Birth", "Transfer", "Transfer"
            ]
        }
        
        source_df = pd.DataFrame(admission_source_data)
        
        # Color code by category
        def color_source_type(val):
            if val == "Referral":
                return 'background-color: #e8f5e8; color: #2e7d32'
            elif val == "Transfer":
                return 'background-color: #e3f2fd; color: #1565c0'
            elif val == "Emergency":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "Birth":
                return 'background-color: #fff3e0; color: #f57c00'
            elif val == "Legal":
                return 'background-color: #f3e5f5; color: #7b1fa2'
            elif val == "Readmission":
                return 'background-color: #fff8e1; color: #f9a825'
            else:
                return 'background-color: #f5f5f5; color: #666'
        
        # Filter by category
        categories = ['All'] + sorted(source_df['Category'].unique().tolist())
        selected_category = st.selectbox("Filter by category:", categories)
        
        if selected_category != 'All':
            filtered_source_df = source_df[source_df['Category'] == selected_category]
        else:
            filtered_source_df = source_df
        
        st.dataframe(
            filtered_source_df.style.applymap(color_source_type, subset=['Category']),
            use_container_width=True,
            hide_index=True
        )
        
        # Source category breakdown
        st.subheader("Admission Source Summary")
        source_counts = source_df['Category'].value_counts()
        
        cols = st.columns(len(source_counts))
        for i, (category, count) in enumerate(source_counts.items()):
            cols[i].metric(category, count)
    
    # Download options
    st.markdown("---")
    st.subheader("📥 Download Reference Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        admission_csv = pd.DataFrame(admission_data).to_csv(index=False)
        st.download_button(
            "Download Admission Types",
            admission_csv,
            "admission_types.csv",
            "text/csv"
        )
    
    with col2:
        discharge_csv = pd.DataFrame(discharge_data).to_csv(index=False)
        st.download_button(
            "Download Discharge Codes", 
            discharge_csv,
            "discharge_codes.csv",
            "text/csv"
        )
    
    with col3:
        source_csv = pd.DataFrame(admission_source_data).to_csv(index=False)
        st.download_button(
            "Download Admission Sources",
            source_csv, 
            "admission_sources.csv",
            "text/csv"
        )
#---------------------------------Contact----------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "📞 Fusion5 Contact":
    st.title("📞 Contact Fusion 5 Team")
    st.markdown("Get in touch with our healthcare AI specialists for support, partnerships, or technical inquiries.")
    
    # Team contact cards
    st.subheader("👥 Team Members")
    
    team_members = [
        {"name": "Siri", "role": "Machine Learning Specialist", "email": "siri@fusion5.com", "focus": "Model Development & AI Research"},
        {"name": "Srividya", "role": "Deployment Engineer", "email": "srividya@fusion5.com", "focus": "System Integration & Cloud Infrastructure"},
        {"name": "Greggy", "role": "Data Scientist", "email": "greggy@fusion5.com", "focus": "Data Pipeline & Analytics"},
        {"name": "Chris", "role": "Business Development", "email": "chris@fusion5.com", "focus": "Healthcare Partnerships & Strategy"},
        {"name": "Sean", "role": "Web Developer", "email": "sean@fusion5.com", "focus": "Frontend Development & User Experience"}
    ]
    
    # Display team members in a grid
    cols = st.columns(2)
    for i, member in enumerate(team_members):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div style="
                    border: 2px solid #4a90c2; 
                    border-radius: 12px; 
                    padding: 1.5rem; 
                    margin: 1rem 0;
                    background-color: #f8fafe;
                    text-align: center;
                ">
                    <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">{member['name']}</h4>
                    <p style="color: #2c5aa0; font-weight: 500; margin-bottom: 0.5rem;">{member['role']}</p>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">{member['focus']}</p>
                    <a href="mailto:{member['email']}?subject=Healthcare AI Inquiry" 
                       style="
                           background-color: #4a90c2; 
                           color: white; 
                           padding: 0.5rem 1rem; 
                           border-radius: 6px; 
                           text-decoration: none; 
                           font-weight: 500;
                           display: inline-block;
                       ">
                        📧 Contact {member['name']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick contact form
    st.subheader("✉️ Send Us a Message")
    st.markdown("Choose a team member and compose your message. This will open your email client with pre-filled information.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Contact selection
        contact_options = [f"{member['name']} - {member['role']}" for member in team_members]
        selected_contact = st.selectbox("Who would you like to contact?", contact_options)
        
        # Get selected member email
        selected_index = contact_options.index(selected_contact)
        selected_email = team_members[selected_index]['email']
        selected_name = team_members[selected_index]['name']
        
        # Message type
        message_types = [
            "General Inquiry",
            "Technical Support", 
            "Partnership Opportunity",
            "Model Performance Question",
            "Data Integration Help",
            "Custom Development Request"
        ]
        
        message_type = st.selectbox("Type of inquiry:", message_types)
    
    with col2:
        # User information
        user_name = st.text_input("Your Name:")
        user_organization = st.text_input("Organization (optional):")
        
        # Message content
        user_message = st.text_area(
            "Your Message:",
            placeholder="Please describe your inquiry or how we can help you...",
            height=120
        )
    
    # Generate email
    if st.button("📧 Open Email Client", use_container_width=True, type="primary"):
        if user_name and user_message:
            # Create email content
            subject = f"Healthcare AI Inquiry - {message_type}"
            
            body = f"""Hello {selected_name},

My name is {user_name}{f' from {user_organization}' if user_organization else ''}.

Type of Inquiry: {message_type}

Message:
{user_message}

Best regards,
{user_name}

---
Sent via Fusion 5 Healthcare AI Platform
"""
            
            # URL encode the email content
            import urllib.parse
            encoded_subject = urllib.parse.quote(subject)
            encoded_body = urllib.parse.quote(body)
            
            # Create mailto link
            mailto_link = f"mailto:{selected_email}?subject={encoded_subject}&body={encoded_body}"
            
            # Display the link
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <a href="{mailto_link}" 
                   style="
                       background-color: #4a90c2; 
                       color: white; 
                       padding: 1rem 2rem; 
                       border-radius: 8px; 
                       text-decoration: none; 
                       font-weight: 500;
                       font-size: 1.1rem;
                       display: inline-block;
                   ">
                    📧 Send Email to {selected_name}
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"Email prepared for {selected_name}! Click the button above to open your email client.")
            
        else:
            st.warning("Please fill in your name and message before generating the email.")
    
    # Alternative contact methods
    st.markdown("---")
    st.subheader("🌐 Alternative Contact Methods")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📧 General Email**  
        [info@fusion5.com](mailto:info@fusion5.com)
        
        For general inquiries and information
        """)
    
    with col2:
        st.markdown("""
        **🆘 Technical Support**  
        [support@fusion5.com](mailto:support@fusion5.com)
        
        For technical issues and troubleshooting
        """)
    
    with col3:
        st.markdown("""
        **🤝 Partnerships**  
        [partnerships@fusion5.com](mailto:partnerships@fusion5.com)
        
        For business development and collaborations
        """)
    
    # FAQ section
    st.markdown("---")
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("How quickly can I expect a response?"):
        st.markdown("""
        - **Technical Support**: Within 24 hours during business days
        - **General Inquiries**: Within 48 hours
        - **Partnership Opportunities**: Within 1 week
        - **Urgent Medical Issues**: Please contact your healthcare provider directly
        """)
    
    with st.expander("What information should I include in my inquiry?"):
        st.markdown("""
        For the best support experience, please include:
        - Your name and organization
        - Specific details about your use case or issue
        - Any error messages or screenshots (if applicable)
        - Your preferred method and timeline for follow-up
        """)
    
    with st.expander("Can you integrate with our existing healthcare systems?"):
        st.markdown("""
        Yes! Our team specializes in healthcare system integration. Contact **Srividya** (Deployment Engineer) 
        or **Chris** (Business Development) to discuss:
        - EHR system integration
        - FHIR compliance
        - Custom API development
        - Cloud deployment options
        """)
    
    # Disclaimer
    st.markdown("---")
    st.info("🏥 **Medical Disclaimer**: This platform is for research and educational purposes. For medical emergencies or urgent patient care decisions, please contact qualified healthcare professionals immediately.")


    
#else:
  #  st.error("Page not found. Please select a valid option from the sidebar.")
