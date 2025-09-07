"""Utility to analyze extracted visual features."""
import os
import sys
from pathlib import Path
import json
import random
import numpy as np
from pprint import pprint

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import PROCESSED_DATA_DIR

def analyze_features(num_samples: int = 3):
    """Analyze random samples of extracted features."""
    features_path = PROCESSED_DATA_DIR / 'visual_features/thumbnail_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    print(f"\nAnalyzing {num_samples} random thumbnails:")
    print("=" * 50)
    
    samples = random.sample(features, num_samples)
    
    for sample in samples:
        print(f"\nThumbnail: {sample['video_id']}")
        print("-" * 30)
        
        # Color features
        colors = np.array(sample['color_features']['dominant_colors'])
        print("\nColor Analysis:")
        print(f"Brightness: {sample['color_features']['brightness']:.2f}")
        print(f"Contrast: {sample['color_features']['contrast']:.2f}")
        print(f"Color Entropy: {sample['color_features']['color_entropy']:.2f}")
        
        # Face features
        print("\nFace Analysis:")
        print(f"Number of faces: {sample['face_features']['num_faces']}")
        if sample['face_features']['faces']:
            for i, face in enumerate(sample['face_features']['faces']):
                print(f"Face {i+1} relative size: {face['relative_size']:.3f}")
                
        # Text features
        print("\nText Analysis:")
        print(f"Has text: {sample['text_features']['has_text']}")
        print(f"Word count: {sample['text_features']['word_count']}")
        if sample['text_features']['text_content']:
            print(f"Text content: {sample['text_features']['text_content'][:100]}...")
            
        # Composition features
        print("\nComposition Analysis:")
        print(f"Edge density: {sample['composition_features']['edge_density']:.3f}")
        print(f"Symmetry - Horizontal: {sample['composition_features']['symmetry']['horizontal']:.3f}")
        print(f"Symmetry - Vertical: {sample['composition_features']['symmetry']['vertical']:.3f}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    analyze_features()
