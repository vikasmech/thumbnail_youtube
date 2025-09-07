"""Module for building the final dataset combining all features."""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_all_features() -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Load and combine all feature sets.
    
    Returns:
        Tuple of (combined DataFrame, feature group mapping)
    """
    # Load performance metrics
    performance_df = pd.read_csv(PROCESSED_DATA_DIR / 'metrics/thumbnail_performance.csv')
    
    # Load visual features
    with open(PROCESSED_DATA_DIR / 'visual_features/enhanced_thumbnail_features.json', 'r') as f:
        visual_features = json.load(f)
    visual_df = pd.json_normalize(visual_features)
    
    # Load text features
    with open(PROCESSED_DATA_DIR / 'text_features/title_features.json', 'r') as f:
        text_features = json.load(f)
    text_df = pd.json_normalize(text_features)
    
    # Merge all features
    df = performance_df.merge(visual_df, on='video_id', how='inner')
    df = df.merge(text_df, on='video_id', how='inner')
    
    # Create feature groups mapping
    feature_groups = {
        'performance_metrics': [
            'views_per_day', 'likes_per_day', 'comments_per_day',
            'views_vs_channel_avg', 'likes_vs_channel_avg', 'comments_vs_channel_avg',
            'engagement_rate'
        ],
        'visual_basic': [
            'color_features.brightness', 'color_features.local_contrast',
            'color_features.color_entropy', 'color_features.warmth_score',
            'face_features.num_faces', 'face_features.total_face_area_ratio'
        ],
        'visual_composition': [
            'composition_features.edge_density',
            'composition_features.visual_complexity',
            'composition_features.thirds_adherence',
            'composition_features.symmetry.horizontal',
            'composition_features.symmetry.vertical',
            'composition_features.symmetry.radial'
        ],
        'text_basic': [
            'basic_features.char_length', 'basic_features.word_length',
            'basic_features.avg_word_length', 'basic_features.non_alnum_chars'
        ],
        'text_sentiment': [
            'sentiment_features.sentiment_compound',
            'sentiment_features.sentiment_pos',
            'sentiment_features.sentiment_neg',
            'sentiment_features.sentiment_neu',
            'sentiment_features.subjectivity'
        ],
        'text_patterns': [
            'pattern_features.number_count',
            'pattern_features.question_count',
            'pattern_features.exclamation_count',
            'pattern_features.emoji_count'
        ]
    }
    
    return df, feature_groups

def prepare_training_data(df: pd.DataFrame, feature_groups: Dict[str, list], 
                         target: str = 'thumbnail_effectiveness',
                         test_size: float = 0.2,
                         random_state: int = 42) -> Dict:
    """Prepare data for model training.
    
    Args:
        df: Combined feature DataFrame
        feature_groups: Dictionary mapping feature group names to column lists
        target: Target variable name
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train/test splits and feature information
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Combine all features
    feature_cols = []
    for group in feature_groups.values():
        feature_cols.extend(group)
    
    # Remove any missing values
    df = df.dropna(subset=feature_cols + [target])
    
    # Split features and target
    X = df[feature_cols]
    y = df[target]
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create feature names mapping
    feature_indices = {}
    start_idx = 0
    for group_name, group_features in feature_groups.items():
        n_features = len(group_features)
        feature_indices[group_name] = {
            'start': start_idx,
            'end': start_idx + n_features,
            'features': group_features
        }
        start_idx += n_features
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'feature_indices': feature_indices,
        'scaler': scaler,
        'original_data': df
    }

def main():
    """Build and save the final dataset."""
    print("Loading features...")
    df, feature_groups = load_all_features()
    
    print("\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\nFeature groups:")
    for group, features in feature_groups.items():
        print(f"\n{group}:")
        for feature in features:
            print(f"- {feature}")
    
    # Save combined dataset
    output_dir = PROCESSED_DATA_DIR / 'modeling'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'combined_features.csv', index=False)
    
    # Save feature groups
    with open(output_dir / 'feature_groups.json', 'w') as f:
        json.dump(feature_groups, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}")

if __name__ == "__main__":
    main()
