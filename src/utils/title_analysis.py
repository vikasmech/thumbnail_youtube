"""Utility to analyze extracted title features."""
import os
import sys
from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from collections import Counter

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import PROCESSED_DATA_DIR

def analyze_title_features(num_samples: int = 3):
    """Analyze random samples of extracted title features."""
    features_path = PROCESSED_DATA_DIR / 'text_features/title_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    print(f"\nAnalyzing {num_samples} random titles:")
    print("=" * 70)
    
    samples = random.sample(features, num_samples)
    
    for sample in samples:
        print(f"\nTitle: {sample['title']}")
        print("-" * 50)
        
        # Basic Features
        basic = sample['basic_features']
        print("\nBasic Features:")
        print(f"Length: {basic['char_length']} chars, {basic['word_length']} words")
        print(f"Average word length: {basic['avg_word_length']:.1f} chars")
        print(f"Non-alphanumeric chars: {basic['non_alnum_chars']}")
        
        # Pattern Features
        patterns = sample['pattern_features']
        print("\nPattern Analysis:")
        for name, count in patterns.items():
            if name.endswith('_count') and count > 0:
                print(f"{name}: {count}")
                
        # Sentiment Analysis
        sentiment = sample['sentiment_features']
        print("\nSentiment Analysis:")
        print(f"Compound score: {sentiment['sentiment_compound']:.3f}")
        print(f"Positive: {sentiment['sentiment_pos']:.3f}")
        print(f"Negative: {sentiment['sentiment_neg']:.3f}")
        print(f"Neutral: {sentiment['sentiment_neu']:.3f}")
        print(f"Subjectivity: {sentiment['subjectivity']:.3f}")
        
        # Linguistic Features
        ling = sample['linguistic_features']
        print("\nLinguistic Analysis:")
        print("POS Ratios:")
        for pos, ratio in ling['pos_ratios'].items():
            print(f"  {pos}: {ratio:.3f}")
        print(f"Content word ratio: {ling['content_ratio']:.3f}")
        print(f"Stop word ratio: {ling['stop_ratio']:.3f}")
        
        # Semantic Features
        semantic = sample['semantic_features']
        print("\nSemantic Analysis:")
        print(f"Clickbait phrases: {semantic['clickbait_count']}")
        print(f"Tech keywords: {semantic['tech_keyword_count']}")
        
        structure = semantic['title_structure']
        print("\nTitle Structure:")
        for key, value in structure.items():
            if value:
                print(f"- {key}")
                
        print("\n" + "=" * 70)

def plot_feature_distributions():
    """Plot distributions of key features."""
    features_path = PROCESSED_DATA_DIR / 'text_features/title_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    # Extract metrics for plotting
    metrics = {
        'Title Length': [f['basic_features']['char_length'] for f in features],
        'Word Count': [f['basic_features']['word_length'] for f in features],
        'Sentiment': [f['sentiment_features']['sentiment_compound'] for f in features],
        'Subjectivity': [f['sentiment_features']['subjectivity'] for f in features]
    }
    
    # Create distribution plots
    plt.figure(figsize=(15, 10))
    for i, (name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        sns.histplot(values, kde=True)
        plt.title(f'Distribution of {name}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = PROCESSED_DATA_DIR / 'text_features/title_distributions.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nFeature distribution plot saved to {plot_path}")
    
def analyze_common_patterns():
    """Analyze common patterns in titles."""
    features_path = PROCESSED_DATA_DIR / 'text_features/title_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    # Collect statistics
    total_titles = len(features)
    pattern_stats = {
        'question_titles': sum(1 for f in features if f['semantic_features']['title_structure']['is_question']),
        'tutorial_titles': sum(1 for f in features if f['semantic_features']['title_structure']['is_tutorial']),
        'comparison_titles': sum(1 for f in features if f['semantic_features']['title_structure']['is_comparison']),
        'numbered_titles': sum(1 for f in features if f['semantic_features']['title_structure']['starts_with_number'])
    }
    
    # Count clickbait and tech keywords
    total_clickbait = sum(f['semantic_features']['clickbait_count'] for f in features)
    total_tech = sum(f['semantic_features']['tech_keyword_count'] for f in features)
    
    print("\nTitle Pattern Analysis:")
    print("=" * 50)
    print(f"\nTotal titles analyzed: {total_titles}")
    
    print("\nTitle Types:")
    for pattern, count in pattern_stats.items():
        percentage = (count / total_titles) * 100
        print(f"{pattern}: {count} ({percentage:.1f}%)")
        
    print(f"\nTotal clickbait phrases: {total_clickbait}")
    print(f"Average clickbait phrases per title: {total_clickbait/total_titles:.2f}")
    
    print(f"\nTotal tech keywords: {total_tech}")
    print(f"Average tech keywords per title: {total_tech/total_titles:.2f}")
    
    # Calculate average sentiment
    avg_sentiment = np.mean([f['sentiment_features']['sentiment_compound'] for f in features])
    avg_subjectivity = np.mean([f['sentiment_features']['subjectivity'] for f in features])
    
    print(f"\nAverage sentiment score: {avg_sentiment:.3f}")
    print(f"Average subjectivity score: {avg_subjectivity:.3f}")

if __name__ == "__main__":
    analyze_title_features()
    plot_feature_distributions()
    analyze_common_patterns()
