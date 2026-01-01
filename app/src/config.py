import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
ASSETS_DIR = BASE_DIR / 'assets'

#MODEL FILES
TFIDF_VECTORIZER_PATH = MODELS_DIR / 'tfidf_vectorizer.pkl'
SENTIMENT_MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
METADATA_PATH = MODELS_DIR / 'model_info.json'

# Aliases for backward compatibility
MODEL_PATH = SENTIMENT_MODEL_PATH
VECTORIZER_PATH = TFIDF_VECTORIZER_PATH

# app settings
APP_TITLE = "Sentiment Analysis Web Application"
APP_ICON = ASSETS_DIR / 'app_icon.png'
PAGE_LAYOUT = 'wide'
INITIAL_SIDEBAR_STATE = 'expanded'

# MODEL SETTINGS
MIN_TEXT_LENGTH = 3 #Longueur minimale du texte accepté.
MAX_TEXT_LENGTH = 1000
HISTORY_LIMIT = 10 #Nombre maximum d’analyses conservées en historique

#UI SETTINGS
SENTIMENT_CONFIG = {
    'Positive': {
        'emoji': '😊',
        'color': '#51cf66',
        'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    'Negative': {
        'emoji': '😢',
        'color':  '#ff6b6b',
        'gradient': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    'Neutral': {
        'emoji': '😐',
        'color': '#4dabf7',
        'gradient':  'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    }
}
# TEXT EXAMPLES

EXAMPLES = {
    "😊 Positive": "I absolutely love this product! It's amazing and exceeded all my expectations.  Highly recommend! ",
    "😢 Negative": "This is the worst experience I've ever had.  Terrible quality and awful customer service.",
    "😐 Neutral": "The meeting is scheduled for tomorrow at 3 PM in conference room B.",
    "🛍️ Product Review": "The phone has great battery life and camera quality, but the price is a bit high for the features.",
    "💼 Business":  "Our Q4 results show steady growth with a 15% increase in revenue compared to last quarter.",
    "🎬 Movie Review": "The cinematography was breathtaking, though the plot felt rushed in the final act."
}
# AUTHOR INFO

AUTHOR = {
    'name': 'Belkhlifi Anass',
    'email': 'anassbelkhlifi76@gmail.com',
    'github':  'https://github.com/anass-belkhlifi',
    'linkedin': 'https://www.linkedin.com/in/anass-belkhlifi-470b88234',
    'university': 'University of IBN TOFAIL',
    'project': 'NLP Mini-Project',
    'date': 'December 2025',
    'description': 'A project focused on sentiment analysis using NLP techniques.'
}