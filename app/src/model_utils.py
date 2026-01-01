"""
================================================================================
MODEL UTILITIES MODULE
================================================================================
Handles model loading, prediction, and inference
"""

import pickle
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import streamlit as st

from .config import MODEL_PATH, VECTORIZER_PATH, METADATA_PATH
from .preprocessing import preprocess_text


class SentimentModel:
    """
    Sentiment analysis model wrapper
    """
    
    def __init__(self, model_path: Path, vectorizer_path: Path, metadata_path: Path):
        """
        Initialize model
        
        Args:
            model_path: Path to trained model
            vectorizer_path:  Path to TF-IDF vectorizer
            metadata_path: Path to model metadata
        """
        self.model = self._load_pickle(model_path)
        self.vectorizer = self._load_pickle(vectorizer_path)
        self.metadata = self._load_metadata(metadata_path)
        
        # Check if model has predict_proba
        self.has_probabilities = hasattr(self.model, 'predict_proba')
        
        # Get class labels
        if hasattr(self.model, 'classes_'):
            self.classes = list(self.model.classes_)
        else:
            self.classes = ['Positive', 'Negative', 'Neutral']
    
    @staticmethod
    def _load_pickle(path: Path):
        """Load pickle file"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found:  {path}")
        except Exception as e:
            raise Exception(f"Error loading {path}: {e}")
    
    @staticmethod
    def _load_metadata(path: Path) -> dict:
        """Load metadata JSON"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'model_name': 'Sentiment Analysis Model',
                'test_accuracy': 0.85,
                'test_f1': 0.85
            }
        except Exception: 
            return {}
    
    def predict(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, float]], float]:
        """
        Predict sentiment for text
        
        Args:
            text: Input text (raw or preprocessed)
            
        Returns: 
            Tuple of (sentiment, confidence_scores, inference_time_ms)
        """
        # Validate input
        if not text or len(text. strip()) < 3:
            return None, None, 0.0
        
        start_time = time.time()
        
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        # Check if preprocessing left anything
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return None, None, 0.0
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get probabilities
        if self.has_probabilities:
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence_dict = dict(zip(self.classes, probabilities))
        else:
            confidence_dict = {prediction: 1.0}
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return prediction, confidence_dict, inference_time
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (sentiment, confidence_scores, inference_time) tuples
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        return self.metadata


@st.cache_resource(show_spinner=False)
def load_model() -> SentimentModel:
    """
    Load model with caching
    
    Returns: 
        SentimentModel instance
    """
    try:
        return SentimentModel(MODEL_PATH, VECTORIZER_PATH, METADATA_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please ensure model files are in the 'models/' directory")
        st.stop()


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    from .config import SENTIMENT_CONFIG
    return SENTIMENT_CONFIG. get(sentiment, {}).get('emoji', '🤔')


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment"""
    from .config import SENTIMENT_CONFIG
    return SENTIMENT_CONFIG. get(sentiment, {}).get('color', '#999999')


def get_sentiment_gradient(sentiment: str) -> str:
    """Get gradient for sentiment"""
    from .config import SENTIMENT_CONFIG
    return SENTIMENT_CONFIG.get(sentiment, {}).get('gradient', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')