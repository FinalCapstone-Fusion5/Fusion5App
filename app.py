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

# --- Streamlit Page Configuration & UI ---
st.set_page_config(page_title="Medicine Feedback Analysis", page_icon="💊", layout="wide")
st.title("💊 Medicine Feedback Analysis Pipeline")
st.markdown("Upload a patient feedback CSV file and select a drug to analyze its sentiment and effectiveness.")

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

