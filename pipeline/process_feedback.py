# -*- coding: utf-8 -*-
"""
This module contains all data processing and feature engineering functions for
the medicine feedback pipeline. It includes functionality to build and save a
scikit-learn compatible preprocessing pipeline, and to use that pipeline for
transforming new data.
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
import os
import ssl
import logging
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# --- NLTK Data Download ---
def download_nltk_data():
    """
    Downloads all necessary NLTK data to a local directory to ensure
    it's found by the script, bypassing system path and SSL certificate issues.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    local_nltk_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(local_nltk_dir):
        os.makedirs(local_nltk_dir)

    if local_nltk_dir not in nltk.data.path:
        nltk.data.path.append(local_nltk_dir)

    resources_to_download = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet',
        'punkt_tab': 'tokenizers/punkt_tab',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng'
    }
    for package_id, path in resources_to_download.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"NLTK resource '{package_id}' not found. Downloading...")
            nltk.download(package_id, download_dir=local_nltk_dir, quiet=True)
            logger.info(f"'{package_id}' downloaded successfully.")

# --- Helper Functions for Preprocessing ---

def preprocess_text(text):
    """Applies a series of advanced preprocessing steps to a given text."""
    if not isinstance(text, str): return ""
    
    standardized_text = text.lower()
    negation_handled_text = re.sub(r'\b(not|no|never)\s', r'\1_', standardized_text)
    tokens = word_tokenize(negation_handled_text)

    stop_words = set(stopwords.words('english'))
    custom_stop_words = {
        'drug', 'medication', 'medicine', 'pill', 'tablet', 'capsule', 'dose',
        'dosage', 'treatment', 'doctor', 'patient', 'prescription', 'day',
        'week', 'month', 'year', 'take', 'took', 'taking', 'use', 'used',
        'using', 'effect', 'side'
    }
    stop_words.update(custom_stop_words)
    stop_words -= {'not', 'no', 'never', 'not_'}
    punct = set(string.punctuation)

    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words and w not in punct]

    pos_tagged = nltk.pos_tag(filtered_tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tagged:
        pos = 'n'
        if tag.startswith('J'): pos = 'a'
        elif tag.startswith('V'): pos = 'v'
        elif tag.startswith('R'): pos = 'r'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))

    return ' '.join(lemmatized_tokens)

def group_rating(rating):
    """Groups a numerical rating into sentiment categories."""
    if rating >= 8: return 2
    elif 4 <= rating < 8: return 1
    else: return 0

def group_effectiveness(effectiveness_str):
    """Groups effectiveness strings into sentiment categories."""
    if not isinstance(effectiveness_str, str): return 1
    effectiveness_str = effectiveness_str.lower()
    if 'highly effective' in effectiveness_str or 'considerably effective' in effectiveness_str: return 2
    elif 'marginally effective' in effectiveness_str: return 1
    elif 'ineffective' in effectiveness_str: return 0
    else: return 1

def group_side_effects(side_effects_str):
    """Groups side effects strings into sentiment categories."""
    if not isinstance(side_effects_str, str): return 1
    side_effects_str = side_effects_str.lower()
    if 'no side effects' in side_effects_str: return 2
    elif 'mild side effects' in side_effects_str: return 1
    elif 'moderate' in side_effects_str or 'severe' in side_effects_str: return 0
    else: return 1

# --- Custom Scikit-learn Transformer ---

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to apply all preprocessing steps to the dataframe.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit method is not needed for this rule-based transformer."""
        return self

    def transform(self, X, y=None):
        """Applies the complete preprocessing logic."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        processed_df = X.copy()

        text_columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[f'{col}_cleaned'] = processed_df[col].apply(preprocess_text)

        if 'rating' in processed_df.columns:
            processed_df['rating_sentiment'] = processed_df['rating'].apply(group_rating)
        if 'effectiveness' in processed_df.columns:
            processed_df['effectiveness_sentiment'] = processed_df['effectiveness'].apply(group_effectiveness)
        if 'sideEffects' in processed_df.columns:
            processed_df['sideeffects_sentiment'] = processed_df['sideEffects'].apply(group_side_effects)

        return processed_df

# --- Pipeline Build and Load Functions ---

def build_and_save_pipeline():
    """
    Builds the scikit-learn pipeline and saves it to a fixed path.
    """
    # Ensure NLTK data is available before building the pipeline
    logger.info("Checking for NLTK resources before building pipeline...")
    download_nltk_data()

    pipeline_path = "./models/process_feedback.pkl"
    models_dir = os.path.dirname(pipeline_path)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("Building the data processing pipeline...")
    data_pipeline = Pipeline(steps=[
        ('preprocessor', DataPreprocessor())
    ])
    
    try:
        with open(pipeline_path, 'wb') as f:
            pickle.dump(data_pipeline, f)
        logger.info(f"Pipeline saved successfully to '{pipeline_path}'")
    except Exception as e:
        logger.error(f"Failed to save pipeline: {e}", exc_info=True)
        raise

def run_preprocessing(df):
    """
    Loads and uses the saved pipeline to process a new DataFrame.
    """
    # Ensure NLTK data is available before attempting to process text
    logger.info("Checking for NLTK resources before running inference...")
    download_nltk_data()

    pipeline_path = "./models/process_feedback.pkl"
    logger.info(f"Loading data processing pipeline from '{pipeline_path}'...")
    try:
        with open(pipeline_path, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        logger.info("Transforming new data with the loaded pipeline...")
        processed_data = loaded_pipeline.transform(df)
        logger.info("Data preprocessing complete.")
        return processed_data

    except FileNotFoundError:
        logger.error(f"Pipeline file not found at '{pipeline_path}'. Please run with --build flag first.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during pipeline inference: {e}", exc_info=True)
        raise

