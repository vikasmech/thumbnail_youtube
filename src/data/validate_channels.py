"""Channel validation script to verify selected channels meet our criteria."""
import os
import sys
from typing import Dict, List

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import json

from config import YOUTUBE_API_KEY, TARGET_CHANNELS, RAW_DATA_DIR

class ChannelValidator:
    """Validates YouTube channels for data collection suitability."""
    
    def __init__(self):
        """Initialize the YouTube API client."""
        if not YOUTUBE_API_KEY:
            raise ValueError("YouTube API key not found in environment variables")
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.validation_results_path = RAW_DATA_DIR / 'channel_validation.json'
        
    def get_channel_stats(self, channel_id: str) -> Dict:
        """Fetch channel statistics and details.
        
        Args:
            channel_id: The YouTube channel ID
            
        Returns:
            Dictionary containing channel statistics
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                print(f"No data found for channel {channel_id}")
                return None
                
            channel = response['items'][0]
            return {
                'channel_id': channel_id,
                'title': channel['snippet']['title'],
                'description': channel['snippet']['description'],
                'subscriber_count': int(channel['statistics']['subscriberCount']),
                'video_count': int(channel['statistics']['videoCount']),
                'view_count': int(channel['statistics']['viewCount']),
                'created_at': channel['snippet']['publishedAt']
            }
            
        except HttpError as e:
            print(f"Error fetching stats for channel {channel_id}: {e}")
            return None

    def get_recent_videos(self, channel_id: str, max_results: int = 10) -> List[Dict]:
        """Fetch recent videos from a channel to analyze upload patterns.
        
        Args:
            channel_id: The YouTube channel ID
            max_results: Maximum number of videos to fetch
            
        Returns:
            List of video data dictionaries
        """
        try:
            request = self.youtube.search().list(
                part="id,snippet",
                channelId=channel_id,
                maxResults=max_results,
                order="date",
                type="video"
            )
            response = request.execute()
            
            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                videos.append({
                    'video_id': video_id,
                    'title': item['snippet']['title'],
                    'published_at': item['snippet']['publishedAt']
                })
            
            return videos
            
        except HttpError as e:
            print(f"Error fetching videos for channel {channel_id}: {e}")
            return []

    def analyze_upload_pattern(self, videos: List[Dict]) -> Dict:
        """Analyze channel upload patterns.
        
        Args:
            videos: List of video data
            
        Returns:
            Dictionary containing upload pattern analysis
        """
        if not videos:
            return {
                'avg_uploads_per_month': 0,
                'consistency_score': 0,
                'is_consistent': False
            }
            
        # Convert dates and sort
        dates = [datetime.strptime(v['published_at'], '%Y-%m-%dT%H:%M:%SZ') 
                for v in videos]
        dates.sort()
        
        # Calculate average uploads per month
        if len(dates) >= 2:
            date_range = dates[-1] - dates[0]
            months = date_range.days / 30.44  # Average days per month
            avg_uploads = len(dates) / months
        else:
            avg_uploads = 0
            
        # Calculate consistency score based on upload intervals
        intervals = []
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1]).days
            intervals.append(interval)
            
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            # Calculate variance in upload schedule
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            # Normalize variance to a 0-1 score (lower variance = higher score)
            consistency_score = 1 / (1 + variance/100)
        else:
            consistency_score = 0
            
        return {
            'avg_uploads_per_month': round(avg_uploads, 2),
            'consistency_score': round(consistency_score, 2),
            'is_consistent': consistency_score > 0.7 and avg_uploads >= 4
        }

    def validate_channels(self) -> Dict:
        """Validate all target channels and generate report.
        
        Returns:
            Dictionary containing validation results for all channels
        """
        validation_results = {}
        
        for channel_id in TARGET_CHANNELS:
            print(f"Validating channel {channel_id}...")
            
            # Get channel statistics
            stats = self.get_channel_stats(channel_id)
            if not stats:
                continue
                
            # Get recent videos
            recent_videos = self.get_recent_videos(channel_id, max_results=10)
            
            # Analyze upload patterns
            upload_analysis = self.analyze_upload_pattern(recent_videos)
            
            # Combine results
            validation_results[channel_id] = {
                **stats,
                **upload_analysis,
                'recent_videos': recent_videos,
                'is_valid': (
                    upload_analysis['is_consistent'] and
                    int(stats['subscriber_count']) > 100000 and
                    int(stats['video_count']) >= 100
                )
            }
        
        # Save validation results
        with open(self.validation_results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
            
        return validation_results

    def print_validation_summary(self, results: Dict):
        """Print a summary of validation results.
        
        Args:
            results: Dictionary containing validation results
        """
        print("\nChannel Validation Summary:")
        print("=" * 80)
        
        for channel_id, data in results.items():
            print(f"\nChannel: {data['title']}")
            print(f"Subscribers: {data['subscriber_count']:,}")
            print(f"Total Videos: {data['video_count']:,}")
            print(f"Avg Monthly Uploads: {data['avg_uploads_per_month']}")
            print(f"Upload Consistency Score: {data['consistency_score']}")
            print(f"Valid for Analysis: {'✓' if data['is_valid'] else '✗'}")
            print("-" * 40)

def main():
    """Run channel validation."""
    validator = ChannelValidator()
    results = validator.validate_channels()
    validator.print_validation_summary(results)

if __name__ == "__main__":
    main()
