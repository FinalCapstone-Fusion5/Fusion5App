#!/usr/bin/env python3
"""
Data Pipeline for Readmission Model
Processes patient_encounters_2023.csv and prepares it for readmission prediction
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ReadmissionDataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.processed_data = None
        
    def load_data(self, file_path):
        """Load the patient encounters data"""
        print(f"Loading data from {file_path}...")
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def preprocess_data(self, data):
        """Preprocess the data for readmission prediction"""
        print("Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Handle age ranges (convert [50-60) to 55, etc.)
        def convert_age_range(age_str):
            if pd.isna(age_str) or age_str == '?':
                return 50  # Default age
            if isinstance(age_str, str) and '[' in age_str and ')' in age_str:
                # Extract numbers from range like [50-60)
                import re
                numbers = re.findall(r'\d+', age_str)
                if len(numbers) >= 2:
                    return (int(numbers[0]) + int(numbers[1])) / 2
                elif len(numbers) == 1:
                    return int(numbers[0])
            return 50  # Default fallback
        
        df['age_numeric'] = df['age'].apply(convert_age_range)
        
        # Handle missing values
        df['number_inpatient'] = df['number_inpatient'].fillna(0)
        df['number_diagnoses'] = df['number_diagnoses'].fillna(0)
        df['time_in_hospital'] = df['time_in_hospital'].fillna(df['time_in_hospital'].median())
        df['num_medications'] = df['num_medications'].fillna(0)
        df['num_lab_procedures'] = df['num_lab_procedures'].fillna(0)
        df['admission_type_id'] = df['admission_type_id'].fillna(1)
        df['num_procedures'] = df['num_procedures'].fillna(0)
        
        # Handle categorical variables
        categorical_cols = ['gender', 'race']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    known_categories = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_categories else 'Unknown')
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Select features for readmission model (matching the trained model)
        feature_cols = [
            'number_inpatient', 'number_diagnoses', 'time_in_hospital', 'age_numeric',
            'num_medications', 'num_lab_procedures', 'admission_type_id', 'num_procedures',
            'gender', 'race'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        self.processed_data = df[feature_cols]
        
        print(f"Preprocessed data shape: {self.processed_data.shape}")
        print(f"Features: {feature_cols}")
        
        return self.processed_data
    
    def fit_scaler(self, data):
        """Fit the scaler on the data (only numerical features)"""
        print("Fitting scaler...")
        
        # Scale only numerical features
        numerical_features = [col for col in self.feature_columns if col not in ['gender', 'race']]
        if numerical_features:
            numerical_data = data[numerical_features]
            self.scaler.fit(numerical_data)
        
        return self.scaler
    
    def transform_data(self, data):
        """Transform data using fitted scaler (only for numerical features)"""
        print("Transforming data...")
        
        # Separate numerical and categorical features
        numerical_features = [col for col in self.feature_columns if col not in ['gender', 'race']]
        categorical_features = [col for col in self.feature_columns if col in ['gender', 'race']]
        
        # Scale only numerical features
        if numerical_features:
            numerical_data = data[numerical_features]
            scaled_numerical = self.scaler.transform(numerical_data)
            
            # Combine scaled numerical and original categorical
            result = data.copy()
            for i, col in enumerate(numerical_features):
                result[col] = scaled_numerical[:, i]
            
            return result.values
        else:
            return data.values
    
    def save_pipeline(self, file_path):
        """Save the pipeline components"""
        pipeline_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to {file_path}")
    
    def load_pipeline(self, file_path):
        """Load the pipeline components"""
        with open(file_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.scaler = pipeline_data['scaler']
        self.label_encoders = pipeline_data['label_encoders']
        self.feature_columns = pipeline_data['feature_columns']
        
        print(f"Pipeline loaded from {file_path}")
    
    def process_file(self, input_file, output_file=None, pipeline_file=None):
        """Complete pipeline: load, preprocess, and save"""
        # Load data
        data = self.load_data(input_file)
        
        # Preprocess
        processed_data = self.preprocess_data(data)
        
        # Fit scaler
        self.fit_scaler(processed_data)
        
        # Transform data
        scaled_data = self.transform_data(processed_data)
        
        # Save processed data
        if output_file:
            processed_df = pd.DataFrame(scaled_data, columns=self.feature_columns)
            processed_df.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")
        
        # Save pipeline
        if pipeline_file:
            self.save_pipeline(pipeline_file)
        
        return scaled_data, processed_data

def main():
    """Main function to run the pipeline"""
    pipeline = ReadmissionDataPipeline()
    
    # Process the data
    input_file = "../healthcare/patient_encounters_2023.csv"
    output_file = "processed_readmission_data_2023.csv"
    pipeline_file = "readmission_data_pipeline.pkl"
    
    try:
        scaled_data, processed_data = pipeline.process_file(
            input_file, output_file, pipeline_file
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Processed data shape: {scaled_data.shape}")
        print(f"Output files created:")
        print(f"  - {output_file}")
        print(f"  - {pipeline_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        print("Please ensure the file exists in the healthcare directory")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
