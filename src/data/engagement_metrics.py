"""Module for calculating engagement metrics and normalized performance scores."""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def calculate_video_age_days(published_at: str) -> float:
    """Calculate video age in days.
    
    Args:
        published_at: ISO format datetime string
        
    Returns:
        Age in days as float
    """
    pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
    now = datetime.now()
    return (now - pub_date).days

def calculate_daily_metrics(video: Dict) -> Dict:
    """Calculate daily engagement metrics.
    
    Args:
        video: Video metadata dictionary
        
    Returns:
        Dictionary with daily metrics
    """
    age_days = max(calculate_video_age_days(video['published_at']), 1)  # Avoid division by zero
    
    return {
        'views_per_day': video['view_count'] / age_days,
        'likes_per_day': video['like_count'] / age_days,
        'comments_per_day': video['comment_count'] / age_days
    }

def calculate_channel_averages(videos: List[Dict]) -> Dict[str, Dict]:
    """Calculate average metrics per channel.
    
    Args:
        videos: List of video metadata dictionaries
        
    Returns:
        Dictionary mapping channel IDs to their average metrics
    """
    channel_metrics = {}
    
    for video in videos:
        channel_id = video['channel_id']
        if channel_id not in channel_metrics:
            channel_metrics[channel_id] = {
                'total_videos': 0,
                'avg_views': 0,
                'avg_likes': 0,
                'avg_comments': 0,
                'avg_views_per_day': 0,
                'avg_likes_per_day': 0,
                'avg_comments_per_day': 0
            }
            
        metrics = channel_metrics[channel_id]
        daily = calculate_daily_metrics(video)
        
        # Update running averages
        metrics['total_videos'] += 1
        n = metrics['total_videos']
        
        metrics['avg_views'] = (metrics['avg_views'] * (n-1) + video['view_count']) / n
        metrics['avg_likes'] = (metrics['avg_likes'] * (n-1) + video['like_count']) / n
        metrics['avg_comments'] = (metrics['avg_comments'] * (n-1) + video['comment_count']) / n
        
        metrics['avg_views_per_day'] = (metrics['avg_views_per_day'] * (n-1) + daily['views_per_day']) / n
        metrics['avg_likes_per_day'] = (metrics['avg_likes_per_day'] * (n-1) + daily['likes_per_day']) / n
        metrics['avg_comments_per_day'] = (metrics['avg_comments_per_day'] * (n-1) + daily['comments_per_day']) / n
    
    return channel_metrics

def calculate_performance_scores(videos: List[Dict]) -> List[Dict]:
    """Calculate normalized performance scores for videos.
    
    Args:
        videos: List of video metadata dictionaries
        
    Returns:
        List of dictionaries with performance metrics
    """
    # Calculate channel averages
    channel_metrics = calculate_channel_averages(videos)
    
    performance_data = []
    
    for video in videos:
        channel_id = video['channel_id']
        channel_avg = channel_metrics[channel_id]
        daily = calculate_daily_metrics(video)
        
        # Calculate relative performance (compared to channel average)
        relative_metrics = {
            'views_vs_avg': daily['views_per_day'] / max(channel_avg['avg_views_per_day'], 1),
            'likes_vs_avg': daily['likes_per_day'] / max(channel_avg['avg_likes_per_day'], 1),
            'comments_vs_avg': daily['comments_per_day'] / max(channel_avg['avg_comments_per_day'], 1)
        }
        
        # Calculate engagement rate
        views = max(video['view_count'], 1)  # Avoid division by zero
        engagement_rate = (video['like_count'] + video['comment_count']) / views
        
        # Calculate thumbnail effectiveness score
        # This is a composite score that considers:
        # 1. Performance relative to channel average
        # 2. Engagement rate
        # 3. Sustained performance over time
        effectiveness_score = (
            relative_metrics['views_vs_avg'] * 0.5 +    # Weight views more heavily
            relative_metrics['likes_vs_avg'] * 0.3 +    # Likes indicate positive engagement
            relative_metrics['comments_vs_avg'] * 0.2    # Comments show strong engagement
        ) * (1 + engagement_rate)  # Multiply by engagement bonus
        
        performance_data.append({
            'video_id': video['video_id'],
            'title': video['title'],
            'channel_id': channel_id,
            'channel_title': video['channel_title'],
            'views': video['view_count'],
            'likes': video['like_count'],
            'comments': video['comment_count'],
            'age_days': calculate_video_age_days(video['published_at']),
            'views_per_day': daily['views_per_day'],
            'likes_per_day': daily['likes_per_day'],
            'comments_per_day': daily['comments_per_day'],
            'views_vs_channel_avg': relative_metrics['views_vs_avg'],
            'likes_vs_channel_avg': relative_metrics['likes_vs_avg'],
            'comments_vs_channel_avg': relative_metrics['comments_vs_avg'],
            'engagement_rate': engagement_rate,
            'thumbnail_effectiveness': effectiveness_score
        })
    
    return performance_data

def main():
    """Calculate and save performance metrics."""
    # Load video metadata
    metadata_path = RAW_DATA_DIR / 'video_metadata.json'
    with open(metadata_path, 'r') as f:
        videos = json.load(f)
    
    # Calculate performance metrics
    performance_data = calculate_performance_scores(videos)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(performance_data)
    
    # Create metrics directory if it doesn't exist
    metrics_dir = PROCESSED_DATA_DIR / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics
    metrics_path = metrics_dir / 'thumbnail_performance.csv'
    df.to_csv(metrics_path, index=False)
    
    # Print summary statistics
    print("\nThumbnail Performance Summary:")
    print("=" * 50)
    print(f"\nTotal videos analyzed: {len(df)}")
    
    print("\nTop 10 Most Effective Thumbnails:")
    top_10 = df.nlargest(10, 'thumbnail_effectiveness')
    for _, row in top_10.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Channel: {row['channel_title']}")
        print(f"Views: {row['views']:,}")
        print(f"Engagement Rate: {row['engagement_rate']:.2%}")
        print(f"Effectiveness Score: {row['thumbnail_effectiveness']:.2f}")
    
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
