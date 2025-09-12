#!/usr/bin/env python3
"""
Data Pipeline for Length of Stay Model
Processes patient_encounters_2023.csv and prepares it for prediction
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LOSDataPipeline:
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
        """Preprocess the data for LOS prediction"""
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
        
        df['age'] = df['age'].apply(convert_age_range)
        
        # Handle weight ranges (convert [125-150) to 137.5, etc.)
        def convert_weight_range(weight_str):
            if pd.isna(weight_str) or weight_str == '?':
                return 70  # Default weight in kg
            if isinstance(weight_str, str) and '[' in weight_str and ')' in weight_str:
                # Extract numbers from range like [125-150)
                import re
                numbers = re.findall(r'\d+', weight_str)
                if len(numbers) >= 2:
                    return (int(numbers[0]) + int(numbers[1])) / 2
                elif len(numbers) == 1:
                    return int(numbers[0])
            return 70  # Default fallback
        
        df['weight'] = df['weight'].apply(convert_weight_range)
        
        # Since there's no height column, we'll estimate BMI using average height
        # Average height: Male ~175cm, Female ~162cm
        df['estimated_height'] = df['gender'].map({'Male': 175, 'Female': 162, 'Unknown': 170})
        df['estimated_height'] = df['estimated_height'].fillna(170)
        
        # Calculate BMI
        df['bmi'] = df['weight'] / ((df['estimated_height'] / 100) ** 2)
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        
        # Handle categorical variables
        categorical_cols = ['gender', 'race', 'admission_type_id', 
                          'discharge_disposition_id', 'admission_source_id']
        
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
        
        # Handle diagnosis columns (diag_1, diag_2, diag_3)
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                # Convert to numeric, handling non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle medication columns (all the diabetes medications)
        med_cols = [col for col in df.columns if col in ['metformin', 'repaglinide', 'nateglinide', 
                    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
                    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
                    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                    'metformin-rosiglitazone', 'metformin-pioglitazone']]
        
        for col in med_cols:
            if col in df.columns:
                # Convert medication responses to binary (Steady/Up/Down = 1, No = 0)
                df[col] = df[col].map({'Steady': 1, 'Up': 1, 'Down': 1, 'No': 0}).fillna(0)
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4])
        df['age_group'] = df['age_group'].astype(int)
        
        # Risk factors
        df['high_risk_age'] = (df['age'] > 65).astype(int)
        df['high_bmi'] = (df['bmi'] > 30).astype(int)
        
        # Count of diagnoses
        df['num_diagnoses'] = df[diag_cols].notna().sum(axis=1)
        
        # Count of medications
        df['num_medications'] = df[med_cols].sum(axis=1)
        
        # Additional features
        df['num_lab_procedures'] = df['num_lab_procedures'].fillna(0)
        df['num_procedures'] = df['num_procedures'].fillna(0)
        df['time_in_hospital'] = df['time_in_hospital'].fillna(df['time_in_hospital'].median())
        
        # Create a combined diagnosis feature (simplified)
        df['diagnosis'] = df[diag_cols].sum(axis=1)
        
        # Map admission_type_id to admission_type for compatibility
        df['admission_type'] = df['admission_type_id']
        
        # Select features for model (matching the trained model)
        feature_cols = [
            'age', 'gender', 'admission_type', 'diagnosis', 
            'num_medications', 'num_procedures', 'num_lab_procedures', 'weight'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        self.processed_data = df[feature_cols]
        
        print(f"Preprocessed data shape: {self.processed_data.shape}")
        print(f"Features: {feature_cols}")
        
        return self.processed_data
    
    def fit_scaler(self, data):
        """Fit the scaler on the data"""
        print("Fitting scaler...")
        self.scaler.fit(data)
        return self.scaler
    
    def transform_data(self, data):
        """Transform data using fitted scaler"""
        print("Transforming data...")
        return self.scaler.transform(data)
    
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
    pipeline = LOSDataPipeline()
    
    # Process the data
    input_file = "../healthcare/patient_encounters_2023.csv"
    output_file = "processed_patient_encounters_2023.csv"
    pipeline_file = "los_data_pipeline.pkl"
    
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
