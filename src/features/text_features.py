"""Module for extracting features from video titles."""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
from collections import Counter
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TitleFeatureExtractor:
    """Extracts features from video titles."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.features_dir = PROCESSED_DATA_DIR / 'text_features'
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common patterns
        self.patterns = {
            'number': r'\d+',
            'question': r'\?',
            'exclamation': r'!',
            'brackets': r'[\[\(\{].*?[\]\)\}]',
            'emoji': r'[\U0001F300-\U0001F9FF]',
            'dollar': r'\$',
            'percentage': r'%',
            'vs': r'\bvs\.?\b|\bversus\b',
            'hashtag': r'#\w+',
            'ellipsis': r'\.{3,}',
            'all_caps_word': r'\b[A-Z]{2,}\b'
        }
        
        # Clickbait phrases
        self.clickbait_phrases = [
            'you won\'t believe',
            'mind blowing',
            'shocking',
            'amazing',
            'incredible',
            'insane',
            'secret',
            'revealed',
            'must see',
            'best ever',
            'worst ever',
            'ultimate',
            'perfect',
            'revolutionary'
        ]
        
        # Tech-specific keywords
        self.tech_keywords = [
            'review', 'tutorial', 'guide', 'vs', 'comparison',
            'unboxing', 'hands-on', 'first look', 'setup',
            'how to', 'tips', 'tricks', 'hack', 'explained',
            'benchmark', 'test', 'analysis', 'overview'
        ]
        
    def extract_basic_features(self, title: str) -> Dict:
        """Extract basic title features.
        
        Args:
            title: Video title string
            
        Returns:
            Dictionary containing basic features
        """
        # Tokenize
        tokens = word_tokenize(title.lower())
        words = [w for w in tokens if w.isalnum()]
        
        # Calculate lengths
        char_length = len(title)
        word_length = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Count non-alphanumeric
        non_alnum = sum(not c.isalnum() and not c.isspace() for c in title)
        
        return {
            'char_length': char_length,
            'word_length': word_length,
            'avg_word_length': float(avg_word_length),
            'non_alnum_chars': non_alnum,
            'char_density': char_length / (word_length if word_length > 0 else 1)
        }
        
    def extract_pattern_features(self, title: str) -> Dict:
        """Extract pattern-based features.
        
        Args:
            title: Video title string
            
        Returns:
            Dictionary containing pattern features
        """
        pattern_counts = {}
        
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, title)
            pattern_counts[f'{name}_count'] = len(matches)
            pattern_counts[f'has_{name}'] = len(matches) > 0
            
        # Additional derived features
        pattern_counts['total_patterns'] = sum(
            count for name, count in pattern_counts.items() 
            if name.endswith('_count')
        )
        
        return pattern_counts
        
    def extract_sentiment_features(self, title: str) -> Dict:
        """Extract sentiment-related features.
        
        Args:
            title: Video title string
            
        Returns:
            Dictionary containing sentiment features
        """
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(title)
        
        # TextBlob sentiment
        blob = TextBlob(title)
        
        return {
            'sentiment_neg': float(vader_scores['neg']),
            'sentiment_neu': float(vader_scores['neu']),
            'sentiment_pos': float(vader_scores['pos']),
            'sentiment_compound': float(vader_scores['compound']),
            'subjectivity': float(blob.sentiment.subjectivity),
            'polarity': float(blob.sentiment.polarity)
        }
        
    def extract_linguistic_features(self, title: str) -> Dict:
        """Extract linguistic features.
        
        Args:
            title: Video title string
            
        Returns:
            Dictionary containing linguistic features
        """
        # POS tagging
        tokens = word_tokenize(title)
        pos_tags = nltk.pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Calculate ratios
        total_tags = len(pos_tags)
        pos_ratios = {
            'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                         pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags,
            'verb_ratio': (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                         pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                         pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags,
            'adj_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                        pos_counts.get('JJS', 0)) / total_tags,
            'adv_ratio': (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                        pos_counts.get('RBS', 0)) / total_tags
        }
        
        # Content word ratio
        content_words = [word for word, tag in pos_tags 
                        if tag.startswith(('NN', 'VB', 'JJ', 'RB'))]
        content_ratio = len(content_words) / total_tags
        
        # Stopword ratio
        stop_ratio = len([w for w in tokens if w.lower() in self.stop_words]) / len(tokens)
        
        return {
            'pos_ratios': pos_ratios,
            'content_ratio': float(content_ratio),
            'stop_ratio': float(stop_ratio)
        }
        
    def extract_semantic_features(self, title: str) -> Dict:
        """Extract semantic features.
        
        Args:
            title: Video title string
            
        Returns:
            Dictionary containing semantic features
        """
        title_lower = title.lower()
        
        # Check for clickbait phrases
        clickbait_count = sum(
            1 for phrase in self.clickbait_phrases 
            if phrase in title_lower
        )
        
        # Check for tech keywords
        tech_keyword_count = sum(
            1 for keyword in self.tech_keywords 
            if keyword in title_lower
        )
        
        # Analyze title structure
        starts_with_number = bool(re.match(r'^\d+', title.strip()))
        starts_with_how = title_lower.startswith('how')
        starts_with_why = title_lower.startswith('why')
        starts_with_what = title_lower.startswith('what')
        
        return {
            'clickbait_count': clickbait_count,
            'tech_keyword_count': tech_keyword_count,
            'clickbait_ratio': clickbait_count / len(title.split()),
            'tech_keyword_ratio': tech_keyword_count / len(title.split()),
            'title_structure': {
                'starts_with_number': starts_with_number,
                'starts_with_how': starts_with_how,
                'starts_with_why': starts_with_why,
                'starts_with_what': starts_with_what,
                'is_question': title.strip().endswith('?'),
                'is_tutorial': 'how to' in title_lower or 'tutorial' in title_lower,
                'is_comparison': 'vs' in title_lower or 'versus' in title_lower,
                'has_year': bool(re.search(r'\b20\d{2}\b', title))
            }
        }
        
    def extract_features(self, video_id: str, title: str) -> Dict:
        """Extract all features for a single title.
        
        Args:
            video_id: YouTube video ID
            title: Video title string
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            features = {
                'video_id': video_id,
                'title': title,
                'basic_features': self.extract_basic_features(title),
                'pattern_features': self.extract_pattern_features(title),
                'sentiment_features': self.extract_sentiment_features(title),
                'linguistic_features': self.extract_linguistic_features(title),
                'semantic_features': self.extract_semantic_features(title)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for video {video_id}: {e}")
            return None
            
    def process_all_titles(self):
        """Extract features from all video titles in the dataset."""
        # Load video metadata
        metadata_path = RAW_DATA_DIR / 'video_metadata.json'
        
        with open(metadata_path, 'r') as f:
            videos = json.load(f)
            
        # Process all titles
        all_features = []
        failed = []
        
        for video in tqdm(videos, desc="Extracting title features"):
            features = self.extract_features(video['video_id'], video['title'])
            
            if features:
                all_features.append(features)
            else:
                failed.append(video['video_id'])
                
        # Save features
        features_path = self.features_dir / 'title_features.json'
        with open(features_path, 'w') as f:
            json.dump(all_features, f, indent=2)
            
        # Print summary
        print(f"\nTitle Feature Extraction Summary:")
        print(f"Successfully processed: {len(all_features)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed videos:")
            for video_id in failed:
                print(f"- {video_id}")
        
        print(f"\nFeatures saved to {features_path}")

if __name__ == "__main__":
    extractor = TitleFeatureExtractor()
    extractor.process_all_titles()
