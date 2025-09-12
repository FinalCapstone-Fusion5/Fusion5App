# -*- coding: utf-8 -*-
"""
This script orchestrates the sentiment analysis pipeline by calling
specialized modules for each step. The --drug_name argument is optional
for inference runs.
"""
import os
import logging
import argparse
import pandas as pd
import sys
import json
from datetime import datetime

# --- Python Path Correction ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------

from pipeline import process_feedback, predict_sentiment, validation

# --- Setup Functions ---

def setup_directories(dir_list):
    """Checks for and creates a list of required directories."""
    for directory in dir_list:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            # This catch is for rare race conditions or permission errors
            logging.error(f"Error creating directory {directory}: {e}")
            raise  # Stop execution if a directory can't be created

def setup_logging(logs_dir):
    """Configures logging to write to a file and the console."""
    log_filename = f"sentiment_pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Logging initialized. Log file at: {log_filepath}")

# --- Main Orchestrator ---
def main(input_filename, drug_name=None, build=False):
    """Main function to orchestrate the sentiment pipeline."""
    # Define required directories
    LOGS_DIR = 'logs/sentiment'
    OUTPUT_DIR = 'output/sentiment'
    #TEST_DATA_DIR = 'test_data'
    MODELS_DIR = 'models'
    
    # Ensure all directories exist before proceeding
    setup_directories([LOGS_DIR, OUTPUT_DIR, MODELS_DIR])
    setup_logging(LOGS_DIR)

    # --- Build Pipeline ---
    if build:
        logging.info("--- Starting Preprocessing Pipeline Build ---")
        process_feedback.build_and_save_pipeline()
        logging.info("--- Preprocessing Pipeline Build Finished ---")
        return

    # Construct the full path to the input file
    full_input_path = os.path.join(input_filename)

    # --- Load and Validate Data ---
    try:
        logging.info(f"Loading data from '{full_input_path}'...")
        input_df = pd.read_csv(full_input_path)
        
        # Validate the dataframe columns for the sentiment pipeline
        if not validation.validate_columns(input_df, 'sentiment'):
            logging.error("Pipeline run aborted due to invalid input file schema.")
            return

    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{full_input_path}'.")
        return
    except Exception as e:
        logging.error(f"A critical error occurred while loading the data: {e}", exc_info=True)
        return

    # --- Execute Sentiment Pipeline ---
    try:
        if drug_name:
            logging.info(f"--- Starting Sentiment Analysis Pipeline for Drug: {drug_name} ---")
        else:
            logging.info("--- Starting Overall Sentiment Analysis Pipeline ---")

        # Step 1: Preprocessing
        processed_df = process_feedback.run_preprocessing(input_df)
        processed_output_path = os.path.join(OUTPUT_DIR, "1_processed_data.csv")
        processed_df.to_csv(processed_output_path, index=False)
        logging.info(f"Intermediate processed data saved to '{processed_output_path}'")

        # Step 2: Prediction and Analysis
        prediction_result = predict_sentiment.run_inference(processed_df, drug_name)
        
        # Save the full dataframe with predictions
        final_df = pd.DataFrame(prediction_result.get("data", []))
        if not final_df.empty:
            final_output_path = os.path.join(OUTPUT_DIR, "2_final_sentiment_data.csv")
            final_df.to_csv(final_output_path, index=False)
            logging.info(f"Final sentiment data saved to '{final_output_path}'")

        # Save the analysis summary
        analysis_summary = prediction_result.get("analysis_summary")
        if analysis_summary:
            summary_output_path = os.path.join(OUTPUT_DIR, "3_analysis_summary.json")
            with open(summary_output_path, 'w') as f:
                json.dump(analysis_summary, f, indent=4)
            logging.info(f"Analysis summary saved to '{summary_output_path}'")

        logging.info("--- Sentiment Analysis Pipeline Finished Successfully ---")

    except Exception:
        logging.error("A critical error occurred in the sentiment pipeline.", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Healthcare Sentiment Analysis Pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input", 
        required=True, 
        help="Filename of the input CSV located in the 'test_data' folder."
    )
    parser.add_argument(
        "--drug_name", 
        type=str, 
        default=None, 
        help="Optional: Name of a specific drug to analyze. If not provided, only overall analysis is performed."
    )
    parser.add_argument(
        '--build', 
        action='store_true', 
        help="If set, builds the preprocessing pipeline. Does not run inference."
    )
    
    args = parser.parse_args()

    main(args.input, args.drug_name, args.build)

