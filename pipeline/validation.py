import logging

logger = logging.getLogger(__name__)

# --- Column Schemas ---
# Define the set of columns required for each pipeline to run successfully.

SENTIMENT_SCHEMA = {
    'urlDrugName',
    'rating',
    'effectiveness',
    'sideEffects',
    'condition',
    'benefitsReview',
    'sideEffectsReview',
    'commentsReview'
}

# This schema is based on the standard dataset used for LOS prediction,
# similar to the one found in your /input folder.
LOS_SCHEMA = {
    'race',
    'gender',
    'age',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'diag_1',
    'diag_2',
    'diag_3',
    'number_diagnoses',
    'max_glu_serum',
    'A1Cresult',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'change',
    'diabetesMed',
    'readmitted'
}

def validate_columns(df, pipeline_type):
    """
    Validates that the input DataFrame contains the required columns for a pipeline.
    This check allows for extra columns in the file, but ensures the core ones exist.

    Args:
        df (pd.DataFrame): The DataFrame loaded from the user's CSV.
        pipeline_type (str): The type of pipeline being run ('sentiment' or 'los').

    Returns:
        bool: True if the file has all required columns, False otherwise.
    """
    logger.info(f"Performing column validation for '{pipeline_type}' pipeline...")
    
    if pipeline_type == 'sentiment':
        expected_cols = SENTIMENT_SCHEMA
    elif pipeline_type == 'los':
        expected_cols = LOS_SCHEMA
    else:
        logger.error(f"Unknown pipeline type '{pipeline_type}' provided for validation.")
        return False

    actual_cols = set(df.columns)
    
    # Check if the set of expected columns is a subset of the actual columns
    if not expected_cols.issubset(actual_cols):
        missing_cols = sorted(list(expected_cols - actual_cols))
        logger.error(f"Input file validation failed. Missing required columns: {missing_cols}")
        return False
        
    logger.info("Column validation successful. All required columns are present.")
    return True
