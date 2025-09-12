# -*- coding: utf-8 -*-
"""
This module loads the trained sentiment model, makes predictions, and performs
a detailed analysis of the results.
"""
import pandas as pd
import warnings
import os
import joblib
import logging
import json
from collections import Counter

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# --- Analysis Helper Functions ---

def analyze_overall_effectiveness(df):
    """
    Analyzes the entire dataframe to find the top/bottom effective drugs and
    the overall sentiment breakdown. Returns the analysis as a dictionary.
    """
    logger.info("\n" + "="*50)
    logger.info("--- Overall Drug Effectiveness Analysis (All Drugs) ---")
    
    analysis_output = {
        "status": "Skipped",
        "top_5_effective": None,
        "bottom_5_effective": None,
        "overall_sentiment_breakdown": {}
    }

    # --- Overall Sentiment Breakdown ---
    if 'predicted_sentiment_str' in df.columns:
        sentiment_counts = {
            "positive": int(df['predicted_sentiment_str'].str.contains("positive", case=False, na=False).sum()),
            "neutral": int(df['predicted_sentiment_str'].str.contains("neutral", case=False, na=False).sum()),
            "negative": int(df['predicted_sentiment_str'].str.contains("negative", case=False, na=False).sum())
        }
        analysis_output["overall_sentiment_breakdown"] = sentiment_counts
        logger.info(f"Overall Sentiment Breakdown: Positive={sentiment_counts['positive']}, Neutral={sentiment_counts['neutral']}, Negative={sentiment_counts['negative']}")
    
    # --- Top/Bottom Drug Effectiveness ---
    if 'effectiveness' not in df.columns or 'urlDrugName' not in df.columns:
        logger.warning("Top/Bottom drug effectiveness analysis skipped: 'effectiveness' or 'urlDrugName' column not found.")
        analysis_output['status'] = "Success (Effectiveness analysis skipped)"
        return analysis_output

    effectiveness_map = {
        'Highly Effective': 5, 'Considerably Effective': 4, 
        'Moderately Effective': 3, 'Marginally Effective': 2, 'Ineffective': 1
    }
    df['effectiveness_score'] = df['effectiveness'].map(effectiveness_map).fillna(0)

    drug_review_counts = df['urlDrugName'].value_counts()
    min_reviews = 3
    reliable_drugs = drug_review_counts[drug_review_counts >= min_reviews].index
    
    if len(reliable_drugs) < 10:
        logger.warning(f"Not enough drugs with sufficient reviews (min {min_reviews}) to generate top/bottom 5 lists.")
        analysis_output["status"] = f"Success (Not enough drugs for top/bottom list)"
        return analysis_output

    filtered_df = df[df['urlDrugName'].isin(reliable_drugs)]
    avg_effectiveness = filtered_df.groupby('urlDrugName')['effectiveness_score'].mean().sort_values(ascending=False)
    
    top_5_drugs = avg_effectiveness.head(5)
    bottom_5_drugs = avg_effectiveness.tail(5)
    
    logger.info(f"\nTop 5 Most Effective Drugs (min {min_reviews} reviews):\n{top_5_drugs.to_string()}")
    logger.info(f"\nBottom 5 Least Effective Drugs (min {min_reviews} reviews):\n{bottom_5_drugs.to_string()}")
    
    analysis_output["status"] = "Success"
    analysis_output["top_5_effective"] = top_5_drugs.to_dict()
    analysis_output["bottom_5_effective"] = bottom_5_drugs.to_dict()
    
    logger.info("="*50 + "\n")
    return analysis_output


def analyze_drug_predictions(df, drug_name):
    """
    Performs a detailed analysis for a specific drug and returns a summary.
    """
    logger.info(f"--- Detailed Analysis for: {drug_name.upper()} ---")
    
    analysis_results = {
        "drug_name": drug_name,
        "status": "Analysis successful",
        "reviews_found": 0,
        "average_rating": "N/A",
        "overall_sentiment": "N/A",
        "sentiment_breakdown": {},
        "top_effective_conditions": {},
        "least_effective_conditions": {},
        "adverse_event_summary": {}
    }

    drug_name_cleaned = drug_name.strip().lower()
    drug_df = df[df['urlDrugName'].str.strip().str.lower() == drug_name_cleaned].copy()
    
    analysis_results["reviews_found"] = len(drug_df)
    logger.info(f"Found {len(drug_df)} reviews for '{drug_name}'.")
    
    if drug_df.empty:
        analysis_results["status"] = f"No data found for drug '{drug_name}'"
        logger.warning(analysis_results["status"])
        return analysis_results
        
    # --- Composite Sentiment Calculation ---
    sentiment_score = -1
    if 'rating_sentiment' in drug_df.columns and not drug_df['rating_sentiment'].empty:
        sentiment_score = drug_df['rating_sentiment'].mean()
        # Average with model prediction if available and numeric
        if 'predicted_sentiment' in drug_df.columns and pd.api.types.is_numeric_dtype(drug_df['predicted_sentiment']):
            model_sentiment_mean = drug_df['predicted_sentiment'].mean(skipna=True)
            if pd.notna(model_sentiment_mean):
                 sentiment_score = (sentiment_score + model_sentiment_mean) / 2

    if sentiment_score > 0.7:
        overall_sentiment = "Positive"
    elif 0.4 <= sentiment_score <= 0.7:
        overall_sentiment = "Neutral"
    elif sentiment_score >= 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Not Determined"

    analysis_results["overall_sentiment"] = overall_sentiment
    logger.info(f"Calculated composite sentiment score: {sentiment_score:.2f} -> {overall_sentiment}")
    
    # --- Other Analyses ---
    if 'predicted_sentiment_str' in drug_df.columns:
        sentiment_counts = {
            "positive": int(drug_df['predicted_sentiment_str'].str.contains("positive", case=False, na=False).sum()),
            "neutral": int(drug_df['predicted_sentiment_str'].str.contains("neutral", case=False, na=False).sum()),
            "negative": int(drug_df['predicted_sentiment_str'].str.contains("negative", case=False, na=False).sum())
        }
        analysis_results["sentiment_breakdown"] = sentiment_counts
        logger.info(f"Drug-specific Sentiment Breakdown: Positive={sentiment_counts['positive']}, Neutral={sentiment_counts['neutral']}, Negative={sentiment_counts['negative']}")

    if 'rating' in drug_df.columns and pd.api.types.is_numeric_dtype(drug_df['rating']):
        avg_rating = drug_df['rating'].mean()
        analysis_results["average_rating"] = round(avg_rating, 2)
        logger.info(f"Average user rating: {analysis_results['average_rating']:.2f} out of 10.")

    if 'effectiveness' in drug_df.columns and 'condition' in drug_df.columns:
        effective_reviews = drug_df[drug_df['effectiveness'].isin(['Highly Effective', 'Considerably Effective'])]
        if not effective_reviews.empty:
            analysis_results["top_effective_conditions"] = effective_reviews['condition'].value_counts().nlargest(2).to_dict()

        ineffective_reviews = drug_df[drug_df['effectiveness'].isin(['Ineffective', 'Marginally Effective'])]
        if not ineffective_reviews.empty:
            analysis_results["least_effective_conditions"] = ineffective_reviews['condition'].value_counts().nlargest(2).to_dict()

    if 'adverse_event_identified' in drug_df.columns:
        adverse_reviews = drug_df[drug_df['adverse_event_identified'] == True]
        count = len(adverse_reviews)
        all_effects = [effect for sublist in adverse_reviews['identified_side_effects'] for effect in sublist]
        effects_counter = Counter(all_effects)

        analysis_results["adverse_event_summary"] = {
            "reviews_with_adverse_events": int(count),
            "identified_side_effects": dict(effects_counter)
        }
        logger.info(f"Adverse event summary: {analysis_results['adverse_event_summary']}")

    return analysis_results

def identify_adverse_events(df):
    adverse_keywords = ['pain', 'nausea', 'headache', 'dizzy', 'anxiety', 'rash', 'vomiting', 'fatigue']
    
    def find_keywords(text):
        if not isinstance(text, str): return []
        return [keyword for keyword in adverse_keywords if keyword in text.lower()]

    review_col = 'sideEffectsReview_cleaned' if 'sideEffectsReview_cleaned' in df.columns else 'sideEffectsReview'
    if review_col in df.columns:
        df['identified_side_effects'] = df[review_col].apply(find_keywords)
        df['adverse_event_identified'] = df['identified_side_effects'].apply(lambda x: len(x) > 0)
    else:
        df['identified_side_effects'] = [[] for _ in range(len(df))]
        df['adverse_event_identified'] = False
    return df

# --- Main Entry Point for this Module ---

def run_inference(inference_df, drug_name=None):
    model_path = "./models/sentiment_model.pkl"
    logger.info(f"--- Starting Sentiment Prediction (Model: {model_path}) ---")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at '{model_path}'.")
        return {"status": "error", "message": "Model file not found."}

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return {"status": "error", "message": f"Error loading model: {e}"}

    df_copy = inference_df.copy()
    
    text_cols = ['benefitsReview_cleaned', 'sideEffectsReview_cleaned', 'commentsReview_cleaned']
    for col in text_cols:
        if col not in df_copy.columns: df_copy[col] = ''
    df_copy[text_cols] = df_copy[text_cols].fillna('')
    df_copy['full_review'] = df_copy[text_cols].agg(' '.join, axis=1)

    logger.info("Predicting sentiment...")
    string_predictions = pipeline.predict(df_copy['full_review'])
    
    df_copy['predicted_sentiment_str'] = pd.Series(string_predictions).str.lower()
    
    sentiment_map_to_numeric = {'positive': 2, 'neutral': 1, 'negative': 0}
    
    df_copy['predicted_sentiment'] = pd.to_numeric(
        df_copy['predicted_sentiment_str'].map(sentiment_map_to_numeric), 
        errors='coerce'
    )
    
    logger.info("Sentiment prediction complete.")
    
    df_with_analysis = identify_adverse_events(df_copy)
    
    # --- Consolidate Analysis Results ---
    analysis_summary = {
        "overall_effectiveness_analysis": analyze_overall_effectiveness(df_with_analysis)
    }
    
    if drug_name:
        analysis_summary["drug_specific_analysis"] = analyze_drug_predictions(df_with_analysis, drug_name)
    
    return {
        "status": "Success",
        "data": df_with_analysis.to_dict(orient='records'),
        "analysis_summary": analysis_summary
    }

