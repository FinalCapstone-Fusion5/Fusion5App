#!/usr/bin/env python3
"""
Improved Batch Readmission Prediction Script
Uses more appropriate threshold based on actual readmission rates.
Redirects procedural output to a log file and writes insights to a JSON file.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
import logging
import json
from datetime import datetime
from pipeline.readmission_data_pipeline import ReadmissionDataPipeline

def setup_logging():
    """Configures logging to write to a timestamped file."""
    log_dir = "logs/readmission"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"prediction_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # Keep logging to console for interactive feedback if needed,
            # but for this request we will primarily use the file.
        ]
    )
    # Add a handler to also print to console for specific messages if desired
    # console_handler = logging.StreamHandler(sys.stdout)
    # logging.getLogger().addHandler(console_handler)

class ImprovedBatchReadmissionPredictor:
    def __init__(self, model_path='models/readmission_model.joblib', pipeline_path='models/readmission_data_pipeline.pkl'):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None
        self.threshold = 0.3  # More appropriate threshold based on actual readmission rate
        self.load_components()
    
    def load_components(self):
        """Load the model and pipeline"""
        logging.info(f"Loading readmission model from {self.model_path}...")
        self.model = pickle.load(open(self.model_path, 'rb'))
        
        logging.info(f"Loading pipeline from {self.pipeline_path}...")
        self.pipeline = ReadmissionDataPipeline()
        self.pipeline.load_pipeline(self.pipeline_path)
        
        logging.info("Components loaded successfully!")
    
    def predict_batch(self, input_file, output_file=None, threshold=None):
        """Predict readmission risk for all records in the input file"""
        if threshold is not None:
            self.threshold = threshold
            
        logging.info(f"Processing {input_file}...")
        logging.info(f"Using threshold: {self.threshold}")
        
        # Load and preprocess data using pipeline
        data = self.pipeline.load_data(input_file)
        processed_data = self.pipeline.preprocess_data(data)
        if output_file:
            try:
                output_dir = "output/readmission"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a dynamic name based on the main output file
                base_name = os.path.basename(output_file)
                file_name_without_ext = os.path.splitext(base_name)[0]
                processed_data_path = os.path.join(output_dir, f"{file_name_without_ext}_preprocessed_data.csv")
                
                # Save the processed dataframe
                processed_data.to_csv(processed_data_path, index=False)
                logging.info(f"Successfully saved post-pipeline processed data to {processed_data_path}")
                
            except Exception as e:
                # Log a warning but don't stop the whole prediction process
                logging.warning(f"Could not save preprocessed data CSV to {processed_data_path}: {e}")
        # --- END: Added code ---
        
        # Make predictions
        logging.info("Making predictions...")
        
        # For CatBoost, we need to specify categorical features by name
        categorical_features = ['gender', 'race']
        
        # Create a Pool object for CatBoost with feature names
        from catboost import Pool
        pool = Pool(processed_data, cat_features=categorical_features)
        
        probabilities = self.model.predict_proba(pool)[:, 1]  # Probability of readmission
        
        # Use custom threshold instead of default 0.5
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Get risk levels and recommendations
        risk_levels = [self._get_risk_level(prob) for prob in probabilities]
        recommendations = [self._get_recommendations(processed_data.iloc[i], prob) 
                          for i, prob in enumerate(probabilities)]
        
        # Create results dataframe
        results = data.copy()
        results['predicted_readmission'] = predictions
        results['readmission_probability'] = np.round(probabilities, 3)
        results['risk_level'] = risk_levels
        results['recommendations'] = recommendations
        
        # Add prediction confidence
        results['prediction_confidence'] = np.maximum(probabilities, 1 - probabilities)
        
        # Save results
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
        
        # Generate, display, and save insights
        self._report_insights(results, output_file)
        
        return results
    
    def _get_risk_level(self, probability):
        """Determine risk level based on readmission probability"""
        if probability < 0.2:
            return "Low"
        elif probability < 0.4:
            return "Medium"
        elif probability < 0.6:
            return "High"
        else:
            return "Very High"
    
    def _get_recommendations(self, patient_data, probability):
        """Generate recommendations based on readmission risk"""
        recommendations = []
        
        if probability > 0.6:
            recommendations.append("High priority discharge planning")
            recommendations.append("Schedule follow-up within 7 days")
        elif probability > 0.4:
            recommendations.append("Enhanced discharge planning")
            recommendations.append("Schedule follow-up within 14 days")
        
        # Use the correct column names from the readmission pipeline
        if patient_data.get('age_numeric', 0) > 65:
            recommendations.append("Geriatric care coordination")
        if patient_data.get('num_medications', 0) > 5:
            recommendations.append("Medication reconciliation review")
        if patient_data.get('number_diagnoses', 0) > 3:
            recommendations.append("Multi-specialty follow-up")
        
        if not recommendations:
            recommendations.append("Standard discharge protocols")
        
        return "; ".join(recommendations)

    def _report_insights(self, results, output_file):
        """Gathers insights, prints them to screen, logs them, and saves as JSON."""
        risk_counts = results['risk_level'].value_counts()
        total_patients = len(results)

        # 1. Gather insights into a dictionary
        insights = {
            "summary": {
                "total_patients_processed": total_patients,
                "patients_predicted_for_readmission": int(results['predicted_readmission'].sum()),
                "predicted_readmission_rate_percent": round((results['predicted_readmission'].sum() / total_patients * 100), 1),
                "average_readmission_probability": round(results['readmission_probability'].mean(), 3),
                "median_readmission_probability": round(results['readmission_probability'].median(), 3),
                "prediction_threshold_used": self.threshold
            },
            "risk_level_distribution": {
                risk: {
                    "count": int(count),
                    "percentage": round((count / total_patients) * 100, 1)
                } for risk, count in risk_counts.items()
            },
            "top_5_highest_risk_patients": results.nlargest(5, 'readmission_probability')[
                ['patient_nbr', 'readmission_probability', 'risk_level', 'age', 'gender']
            ].to_dict(orient='records')
        }

        # 2. Save insights to JSON file
        if output_file:
            self._write_insights_to_json(insights, output_file)

        # 3. Format insights for display and logging
        summary_str = self._format_insights_for_display(insights)

        # 4. Print to screen AND log to file
        print(summary_str)
        logging.info("--- INSIGHTS SUMMARY ---")
        logging.info(summary_str.replace("=", "-")) # Use different separator for log clarity
        logging.info("--- END INSIGHTS SUMMARY ---")
    
    def _write_insights_to_json(self, insights, csv_output_file):
        """Saves the insights dictionary to a JSON file."""
        output_dir = "output/readmission"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(csv_output_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        json_path = os.path.join(output_dir, f"{file_name_without_ext}_insights.json")
        
        try:
            with open(json_path, 'w') as f:
                json.dump(insights, f, indent=4)
            logging.info(f"Insights successfully saved to {json_path}")
        except Exception as e:
            logging.error(f"Failed to write insights to {json_path}: {e}")

    def _format_insights_for_display(self, insights):
        """Formats the insights dictionary into a human-readable string."""
        s = insights["summary"]
        header = "\n" + "="*50 + "\nREADMISSION PREDICTION INSIGHTS\n" + "="*50
        summary_section = (
            f"\nTotal patients processed: {s['total_patients_processed']}\n"
            f"Patients predicted for readmission: {s['patients_predicted_for_readmission']}\n"
            f"Readmission rate: {s['predicted_readmission_rate_percent']}%\n"
            f"Average readmission probability: {s['average_readmission_probability']}\n"
            f"Median readmission probability: {s['median_readmission_probability']}\n"
            f"Threshold used: {s['prediction_threshold_used']}\n"
        )

        risk_section = "\nRisk Level Distribution:\n"
        for risk, data in insights["risk_level_distribution"].items():
            risk_section += f"  {risk}: {data['count']} patients ({data['percentage']}%)\n"

        top_patients_section = "\nTop 5 Highest Risk Patients:\n"
        for patient in insights["top_5_highest_risk_patients"]:
            top_patients_section += (f"  Patient {patient['patient_nbr']}: "
                                     f"{patient['readmission_probability']:.3f} probability "
                                     f"({patient['risk_level']} risk, {patient['age']} {patient['gender']})\n")
        
        return header + summary_section + risk_section + top_patients_section

def main():
    """Main function"""
    # Setup logging first
    setup_logging()

    if len(sys.argv) < 2:
        usage_msg = ("Usage: python3 predict_readmission_pipeline.py <input_file> [output_file] [threshold]\n"
                     "Example: python3 predict_readmission_pipeline.py data/patient_encounters.csv output/readmission/predictions.csv 0.3")
        print(usage_msg) # Print usage to console directly
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/readmission/readmission_predictions.csv"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    model_path='models/readmission_model.joblib'
    pipeline_path='models/readmission_data_pipeline.pkl'
    
    # Check if files exist
    if not os.path.exists(input_file):
        error_msg = f"Error: Input file {input_file} not found"
        logging.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(model_path):
        error_msg = f"Error: {model_path} not found. Please ensure the model file exists."
        logging.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(pipeline_path):
        error_msg = f"Error: {pipeline_path} not found. Please run the data pipeline script first."
        logging.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    try:
        predictor = ImprovedBatchReadmissionPredictor(model_path, pipeline_path)
        predictor.predict_batch(input_file, output_file, threshold)
        
        success_msg = f"\nBatch readmission prediction completed successfully!\n" \
                      f"-> Results CSV saved to: {output_file}\n" \
                      f"-> Insights JSON saved to: output/readmission/ directory\n" \
                      f"-> Detailed logs saved to: logs/readmission/ directory"
        print(success_msg)
        
    except Exception as e:
        error_msg = f"An error occurred during batch prediction: {e}"
        logging.error(error_msg, exc_info=True) # Log full traceback
        print(error_msg, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()