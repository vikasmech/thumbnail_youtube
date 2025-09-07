"""Utility to analyze enhanced visual features."""
import os
import sys
from pathlib import Path
import json
import random
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import PROCESSED_DATA_DIR

def analyze_enhanced_features(num_samples: int = 3):
    """Analyze random samples of extracted enhanced features."""
    features_path = PROCESSED_DATA_DIR / 'visual_features/enhanced_thumbnail_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    print(f"\nAnalyzing {num_samples} random thumbnails:")
    print("=" * 70)
    
    samples = random.sample(features, num_samples)
    
    for sample in samples:
        print(f"\nThumbnail: {sample['video_id']}")
        print("-" * 50)
        
        # Color Analysis
        print("\nColor Analysis:")
        print(f"Color Harmony Score: {sample['color_features']['complementary_harmony']:.3f}")
        print(f"Warmth Score: {sample['color_features']['warmth_score']:.3f}")
        print(f"Local Contrast: {sample['color_features']['local_contrast']:.3f}")
        print(f"Color Confidence: {sample['color_features']['color_confidence']:.3f}")
        
        # Print color distribution by zones
        print("\nColor Distribution by Zones (RGB):")
        zones = sample['color_features']['color_zones']
        for zone, colors in zones.items():
            print(f"{zone.capitalize()}: R={colors[0]:.1f}, G={colors[1]:.1f}, B={colors[2]:.1f}")
            
        # Face Analysis
        print("\nEnhanced Face Analysis:")
        faces = sample['face_features']['faces']
        if faces:
            print(f"Number of faces: {len(faces)}")
            print(f"Total face area ratio: {sample['face_features']['total_face_area_ratio']:.3f}")
            print(f"Face clustering: {sample['face_features']['face_clustering']:.3f}")
            print(f"Average face brightness: {sample['face_features']['avg_face_brightness']:.1f}")
            print(f"Average face contrast: {sample['face_features']['avg_face_contrast']:.1f}")
            
            for i, face in enumerate(faces):
                print(f"\nFace {i+1}:")
                print(f"  Relative size: {face['relative_size']:.3f}")
                print(f"  Aspect ratio: {face['aspect_ratio']:.2f}")
                print(f"  Number of eyes detected: {face['num_eyes']}")
                print(f"  Distance to thirds: {face['thirds_distance']:.1f}")
        else:
            print("No faces detected")
            
        # Composition Analysis
        comp = sample['composition_features']
        print("\nEnhanced Composition Analysis:")
        print(f"Visual Complexity: {comp['visual_complexity']:.3f}")
        print(f"Rule of Thirds Adherence: {comp['thirds_adherence']:.3f}")
        
        print("\nBalance Metrics:")
        print(f"Horizontal: {comp['balance']['horizontal']:.3f}")
        print(f"Vertical: {comp['balance']['vertical']:.3f}")
        
        print("\nSymmetry Analysis:")
        print(f"Horizontal: {comp['symmetry']['horizontal']:.3f}")
        print(f"Vertical: {comp['symmetry']['vertical']:.3f}")
        print(f"Radial: {comp['symmetry']['radial']:.3f}")
        
        if comp['focal_points']:
            print(f"\nFocal Points: {len(comp['focal_points'])} detected")
            for i, point in enumerate(comp['focal_points']):
                print(f"  Point {i+1}: position={point['position']}, saliency={point['saliency']:.3f}")
                
        print("\n" + "=" * 70)
        
def plot_feature_distributions():
    """Plot distributions of key features across all thumbnails."""
    features_path = PROCESSED_DATA_DIR / 'visual_features/enhanced_thumbnail_features.json'
    
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    # Extract key metrics
    metrics = {
        'Color Harmony': [f['color_features']['complementary_harmony'] for f in features],
        'Warmth Score': [f['color_features']['warmth_score'] for f in features],
        'Visual Complexity': [f['composition_features']['visual_complexity'] for f in features],
        'Thirds Adherence': [f['composition_features']['thirds_adherence'] for f in features]
    }
    
    # Create distribution plots
    plt.figure(figsize=(15, 10))
    for i, (name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        sns.histplot(values, kde=True)
        plt.title(f'Distribution of {name}')
        plt.xlabel('Score')
        plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = PROCESSED_DATA_DIR / 'visual_features/feature_distributions.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nFeature distribution plot saved to {plot_path}")

if __name__ == "__main__":
    analyze_enhanced_features()
    plot_feature_distributions()
