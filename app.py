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

# --- Core Pipeline Logic ---
@st.cache_data
def execute_pipeline(input_df, drug_name):
    """
    Orchestrates the data processing and sentiment analysis pipeline.
    """
    log_file = setup_logging()
    logging.info(f"Pipeline run initiated for drug: '{drug_name}'. Log file: {log_file}")

    # Step 1: Process the raw data
    logging.info("Step 1: Preprocessing data...")
    processed_df = process_feedback.run_preprocessing(input_df)
    
    # Step 2: Run sentiment prediction and analysis
    logging.info("Step 2: Predicting sentiment and running analysis...")
    prediction_result = predict_sentiment.run_inference(processed_df, drug_name=drug_name)
    
    logging.info("Pipeline execution complete.")
    return prediction_result, log_file

    # Step 3: Run readmission prediction
    logging.info("Step 3: Predicting readmission...")
    readmission_result = predict_readmission_pipeline.run_inference(processed_df, drug_name=drug_name)

    # Step 4: Run length of stay prediction  
    logging.info("Step 4: Predicting length of stay...")
    lengthofstay_result = predict_lengthofstay_pipeline.run_inference(processed_df, drug_name=drug_name)

# --- Streamlit Page Configuration & UI ---
st.set_page_config(page_title="Medicine Feedback Analysis", page_icon="💊", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🏥 ML Models")
    page = st.radio(
        "Choose Analysis Type:",
        ["🏠 Home", "😊 Sentiment Analysis", "🏥 Readmission Prediction", 
         "⏱️ Length of Stay", "🧠 Retinal CNN Model", "🧪 Retinal Image Test", "Link to Patient Feedback", "Link to Medicine Feedback", "Link to CLinical Codes"]
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
    # Add overview content here
#Readmissions    
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
                    # Read the uploaded file
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Create pipeline instance and process
                    from readmission_data_pipeline import ReadmissionDataPipeline
                    pipeline = ReadmissionDataPipeline()
                    
                    # Process the data
                    processed_data = pipeline.preprocess_data(input_df)
                    pipeline.fit_scaler(processed_data)
                    scaled_data = pipeline.transform_data(processed_data)
                    
                    st.success("✅ Readmission analysis complete!")
                    
                    # Display results (add your prediction logic here)
                    st.subheader("📊 Readmission Analysis Results")
                    st.write(f"Processed {len(input_df)} patient records")
                    st.dataframe(processed_data.head())
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    else:
        st.warning("Please upload a patient encounters CSV file to begin.")

#LOS
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

#CNN

elif page == "🧠 CNN Model":
    st.title("🧠 CNN Eye Image Classification")
    st.markdown("Upload a retinal image to classify diabetic retinopathy severity using our deep learning model.")
    
    # Model info
    st.info("**Model:** Diabetic Retinopathy Classification (5 classes: No DR, Mild, Moderate, Severe, Proliferative)")
    
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
        
        # Sample images info
        st.markdown("---")
        st.markdown("**Expected Image Format:** JPG, JPEG, or PNG retinal/fundus photographs")
        st.markdown("**Model Input:** 512x512 pixel images (automatically resized)")
# --- UI Components ---
with st.sidebar:
    st.header("⚙️ Analysis Configuration")
    uploaded_file = st.file_uploader("1. Upload Patient Feedback CSV", type=['csv'])
    
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
        "2. Select Drug Name for Analysis",
        options=st.session_state.drug_options,
        index=0,
        disabled=(not st.session_state.drug_options)
    )

# Main panel for results
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
                
                # --- Display Results ---
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

