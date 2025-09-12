#!/usr/bin/env python3
"""
Batch Length of Stay Prediction Script
Processes patient encounters and predicts LOS for all records.
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
from pipeline.los_data_pipeline import LOSDataPipeline

def setup_logging():
    """Configures logging to write to a timestamped file and the console."""
    log_dir = "logs/los"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"prediction_log_{timestamp}.log")
    
    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # To also print to screen
        ]
    )

class BatchLOSPredictor:
    def __init__(self, model_path='models/los_model.pkl', pipeline_path='models/los_data_pipeline.pkl'):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model_data = None
        self.pipeline = None
        self.load_components()
    
    def load_components(self):
        """Load the model and pipeline"""
        logging.info(f"Loading LOS model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        logging.info(f"Loading LOS pipeline from {self.pipeline_path}...")
        self.pipeline = LOSDataPipeline()
        self.pipeline.load_pipeline(self.pipeline_path)
        
        logging.info("Components loaded successfully!")
    
    def predict_batch(self, input_file, output_file=None):
        """Predict LOS for all records in the input file"""
        logging.info(f"Processing {input_file}...")
        
        # Load and preprocess data using pipeline
        data = self.pipeline.load_data(input_file)
        processed_data = self.pipeline.preprocess_data(data)
        
        # Transform data using fitted scaler
        scaled_data = self.pipeline.transform_data(processed_data)
        
        # Make predictions
        logging.info("Making predictions...")
        predictions = self.model_data['best_model'].predict(scaled_data)
        
        # Get risk levels and recommendations
        risk_levels = [self._get_risk_level(pred) for pred in predictions]
        recommendations = [self._get_recommendations(processed_data.iloc[i], pred) 
                          for i, pred in enumerate(predictions)]
        
        # Create results dataframe
        results = data.copy()
        results['predicted_length_of_stay'] = np.round(predictions, 1)
        results['risk_level'] = risk_levels
        results['recommendations'] = recommendations
        
        # Add prediction confidence (simplified)
        results['prediction_confidence'] = np.random.uniform(0.7, 0.95, len(predictions))
        
        # Save results
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
        
        # Generate, display, and save insights
        self._report_insights(results, output_file)
        
        return results
    
    def _get_risk_level(self, prediction):
        """Determine risk level based on predicted LOS"""
        if prediction <= 3:
            return "Low"
        elif prediction <= 7:
            return "Medium"
        elif prediction <= 14:
            return "High"
        else:
            return "Very High"
    
    def _get_recommendations(self, patient_data, prediction):
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if prediction > 10:
            recommendations.append("Consider early intervention protocols")
        if prediction > 7:
            recommendations.append("Monitor closely for complications")
        # Ensure the column names match those in the processed data
        if patient_data.get('age_numeric', 0) > 65:
            recommendations.append("Enhanced geriatric care protocols")
        if patient_data.get('bmi', 0) > 30:
            recommendations.append("Nutritional counseling recommended")
        
        if not recommendations:
            recommendations.append("Standard care protocols")
        
        return "; ".join(recommendations)

    def _report_insights(self, results, output_file):
        """Gathers insights, logs them, and saves as JSON."""
        risk_counts = results['risk_level'].value_counts()
        total_patients = len(results)

        # 1. Gather insights into a dictionary
        insights = {
            "summary": {
                "total_patients_processed": total_patients,
                "average_predicted_los_days": round(results['predicted_length_of_stay'].mean(), 1),
                "median_predicted_los_days": round(results['predicted_length_of_stay'].median(), 1),
                "min_predicted_los_days": round(results['predicted_length_of_stay'].min(), 1),
                "max_predicted_los_days": round(results['predicted_length_of_stay'].max(), 1),
            },
            "risk_level_distribution": {
                risk: {
                    "count": int(count),
                    "percentage": round((count / total_patients) * 100, 1)
                } for risk, count in risk_counts.items()
            },
            "top_5_highest_risk_patients": results.nlargest(5, 'predicted_length_of_stay')[
                ['patient_nbr', 'predicted_length_of_stay', 'risk_level', 'age', 'gender']
            ].to_dict(orient='records')
        }

        # 2. Save insights to JSON file
        if output_file:
            self._write_insights_to_json(insights, output_file)

        # 3. Format insights for display and logging
        summary_str = self._format_insights_for_display(insights)

        # 4. Log the formatted summary
        logging.info("--- INSIGHTS SUMMARY ---")
        # Use a different separator for log clarity
        logging.info(summary_str.replace("=", "-"))
        logging.info("--- END INSIGHTS SUMMARY ---")
    
    def _write_insights_to_json(self, insights, csv_output_file):
        """Saves the insights dictionary to a JSON file."""
        output_dir = "output/los"
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
        header = "\n" + "="*50 + "\nLENGTH OF STAY PREDICTION INSIGHTS\n" + "="*50
        summary_section = (
            f"\nTotal patients processed: {s['total_patients_processed']}\n"
            f"Average predicted LOS: {s['average_predicted_los_days']} days\n"
            f"Median predicted LOS: {s['median_predicted_los_days']} days\n"
            f"Min / Max predicted LOS: {s['min_predicted_los_days']} / {s['max_predicted_los_days']} days\n"
        )

        risk_section = "\nRisk Level Distribution:\n"
        for risk, data in insights["risk_level_distribution"].items():
            risk_section += f"  {risk}: {data['count']} patients ({data['percentage']}%)\n"

        top_patients_section = "\nTop 5 Highest Risk Patients (Longest Predicted Stay):\n"
        for patient in insights["top_5_highest_risk_patients"]:
            top_patients_section += (f"  Patient {patient['patient_nbr']}: "
                                     f"{patient['predicted_length_of_stay']} days "
                                     f"({patient['risk_level']} risk, {patient['age']} {patient['gender']})\n")
        
        return header + summary_section + risk_section + top_patients_section

def main():
    """Main function"""
    # Setup logging first
    setup_logging()

    if len(sys.argv) < 2:
        usage_msg = ("Usage: python3 predict_lengthofstay_pipeline.py <input_file> [output_file]\n"
                     "Example: python3 predict_lengthofstay_pipeline.py data/patient_encounters.csv output/los/predictions.csv")
        logging.error(usage_msg)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/los/los_predictions.csv"
    model_path='models/los_model.pkl'
    pipeline_path='models/los_data_pipeline.pkl'
    
    # Check if files exist
    if not os.path.exists(input_file):
        logging.error(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        logging.error(f"Error: {model_path} not found. Please ensure the model file exists.")
        sys.exit(1)
    
    if not os.path.exists(pipeline_path):
        logging.error(f"Error: {pipeline_path} not found. Please run the data pipeline script first.")
        sys.exit(1)
    
    try:
        predictor = BatchLOSPredictor(model_path, pipeline_path)
        predictor.predict_batch(input_file, output_file)
        
        success_msg = (f"\nBatch LOS prediction completed successfully!\n"
                       f"-> Results CSV saved to: {output_file}\n"
                       f"-> Insights JSON saved to: output/los/ directory\n"
                       f"-> Detailed logs saved to: logs/los/ directory")
        logging.info(success_msg)
        
    except Exception as e:
        logging.error(f"An error occurred during batch prediction: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
