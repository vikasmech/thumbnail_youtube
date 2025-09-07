"""Utility to analyze dataset statistics."""
import os
import sys
from pathlib import Path
import json
import pandas as pd
from collections import Counter

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR

def analyze_dataset():
    """Analyze the collected dataset."""
    # Load metadata
    metadata_path = RAW_DATA_DIR / 'video_metadata.json'
    with open(metadata_path, 'r') as f:
        videos = json.load(f)
        
    # Convert to DataFrame
    df = pd.DataFrame(videos)
    
    # Basic stats
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Total videos: {len(df)}")
    
    # Channel distribution
    channel_counts = df['channel_title'].value_counts()
    print("\nVideos per Channel:")
    for channel, count in channel_counts.items():
        print(f"{channel}: {count}")
        
    # View statistics
    views = pd.to_numeric(df['view_count'])
    print("\nView Count Statistics:")
    print(f"Mean: {views.mean():,.0f}")
    print(f"Median: {views.median():,.0f}")
    print(f"Min: {views.min():,.0f}")
    print(f"Max: {views.max():,.0f}")
    
    # Engagement statistics
    likes = pd.to_numeric(df['like_count'])
    comments = pd.to_numeric(df['comment_count'])
    
    print("\nEngagement Statistics:")
    print(f"Average likes: {likes.mean():,.0f}")
    print(f"Average comments: {comments.mean():,.0f}")
    
    # Check thumbnails
    thumbnails_dir = RAW_DATA_DIR / 'thumbnails'
    thumbnail_files = list(thumbnails_dir.glob('*.jpg'))
    print(f"\nThumbnail files: {len(thumbnail_files)}")
    
    # Calculate missing thumbnails
    video_ids = set(df['video_id'])
    thumbnail_ids = {f.stem for f in thumbnail_files}
    missing = video_ids - thumbnail_ids
    
    if missing:
        print("\nMissing thumbnails:")
        for video_id in missing:
            print(f"- {video_id}")

if __name__ == "__main__":
    analyze_dataset()
