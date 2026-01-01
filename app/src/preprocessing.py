import re 
import nltk
from typing import Optional
#Handles all text cleaning and preprocessing operations

#download NLTK data

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))

class TextPreprocessor:
    """
    Professional text preprocessing class with comprehensive cleaning methods
    """
    
    def __init__(self, stop_words: Optional[set] = None, min_word_length: int = 3):
        """
        Initialize preprocessor
        
        Args:
            stop_words: Set of stopwords to remove (default:  NLTK English)
            min_word_length:  Minimum word length to keep
        """
        self.stop_words = stop_words or STOP_WORDS
        self. min_word_length = min_word_length
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.rt_pattern = re.compile(r'\bRT\b', flags=re.IGNORECASE)
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def remove_urls(self, text: str) -> str:
        """Remove all URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text"""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text:  str) -> str:
        """Remove #hashtags from text"""
        return self.hashtag_pattern.sub('', text)
    
    def remove_retweet_markers(self, text: str) -> str:
        """Remove RT markers"""
        return self.rt_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis and emoticons"""
        return self.emoji_pattern.sub('', text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove all numbers"""
        return self.number_pattern.sub('', text)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters, keep only letters and spaces"""
        return self.special_char_pattern.sub('', text)
    
    def normalize_whitespace(self, text:  str) -> str:
        """Normalize whitespace (remove extra spaces, trim)"""
        text = self.whitespace_pattern. sub(' ', text)
        return text.strip()
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_stopwords(self, text:  str) -> str:
        """Remove stopwords and short words"""
        words = text. split()
        filtered = [
            word for word in words 
            if word not in self.stop_words and len(word) >= self.min_word_length
        ]
        return ' '.join(filtered)
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args: 
            text: Raw input text
            
        Returns: 
            Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove URLs
        text = self.remove_urls(text)
        
        # Step 2: Remove mentions
        text = self.remove_mentions(text)
        
        # Step 3: Remove hashtags
        text = self.remove_hashtags(text)
        
        # Step 4: Remove RT markers
        text = self.remove_retweet_markers(text)
        
        # Step 5: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 6: Convert to lowercase
        text = self. to_lowercase(text)
        
        # Step 7: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 8: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 9: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 10: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Final cleanup
        text = self.normalize_whitespace(text)
        
        return text
    
    def get_preprocessing_stats(self, original:  str, processed: str) -> dict:
        """
        Get statistics about preprocessing
        
        Args:
            original: Original text
            processed:  Processed text
            
        Returns: 
            Dictionary with statistics
        """
        return {
            'original_length':  len(original),
            'processed_length': len(processed),
            'original_words': len(original.split()),
            'processed_words':  len(processed.split()),
            'reduction_chars': len(original) - len(processed),
            'reduction_percent': ((len(original) - len(processed)) / len(original) * 100) if len(original) > 0 else 0
        }


# Create global preprocessor instance
preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Convenience function for preprocessing
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    return preprocessor.preprocess(text)