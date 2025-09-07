"""YouTube data collection module."""
import os
import sys
from typing import Dict, List, Optional
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data.thumbnail_manager import ThumbnailManager

from config import (
    YOUTUBE_API_KEY,
    RAW_DATA_DIR,
    TARGET_CHANNELS,
    VIDEOS_PER_CHANNEL
)

class YouTubeDataCollector:
    """Handles collection of YouTube video data and thumbnails."""
    
    def __init__(self):
        """Initialize the YouTube API client and thumbnail manager."""
        if not YOUTUBE_API_KEY:
            raise ValueError("YouTube API key not found in environment variables")
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.metadata_path = RAW_DATA_DIR / 'video_metadata.json'
        self.thumbnail_manager = ThumbnailManager()

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """Fetch videos from a specific channel.
        
        Args:
            channel_id: The YouTube channel ID
            max_results: Maximum number of videos to fetch
            
        Returns:
            List of video data dictionaries
        """
        videos = []
        next_page_token = None
        retries = 3
        
        try:
            # First get channel details
            channel_request = self.youtube.channels().list(
                part="statistics,snippet",
                id=channel_id
            )
            channel_response = channel_request.execute()
            
            if not channel_response.get('items'):
                print(f"Channel {channel_id} not found")
                return videos
                
            channel_info = channel_response['items'][0]
            total_videos = int(channel_info['statistics']['videoCount'])
            print(f"\nChannel: {channel_info['snippet']['title']}")
            print(f"Total videos available: {total_videos}")
            
            # Try different ordering strategies
            for order_method in ['viewCount', 'date', 'rating']:
                if len(videos) >= max_results:
                    break
                    
                print(f"Fetching by {order_method}...")
                page_token = None
                attempt = 0
                
                while len(videos) < max_results and attempt < retries:
                    try:
                        request = self.youtube.search().list(
                            part="id,snippet",
                            channelId=channel_id,
                            maxResults=min(100, max_results - len(videos)),
                            order=order_method,
                            type="video",
                            pageToken=page_token,
                            publishedAfter="2020-01-01T00:00:00Z"  # Recent videos only
                        )
                        response = request.execute()
                        
                        # Track seen video IDs to avoid duplicates
                        seen_videos = {v['video_id'] for v in videos}
                        
                        for item in response['items']:
                            video_id = item['id']['videoId']
                            if video_id not in seen_videos:
                                video_data = self.get_video_details(video_id)
                                if video_data:
                                    videos.append(video_data)
                                    seen_videos.add(video_id)
                        
                        page_token = response.get('nextPageToken')
                        if not page_token:
                            break
                            
                    except Exception as e:
                        print(f"Error on attempt {attempt + 1}: {e}")
                        attempt += 1
                        continue
                    
        except HttpError as e:
            print(f"Error fetching videos for channel {channel_id}: {e}")
        
        return videos

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Fetch detailed information for a specific video.
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            Dictionary containing video details or None if error occurs
        """
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return None
                
            video = response['items'][0]
            return {
                'video_id': video_id,
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'published_at': video['snippet']['publishedAt'],
                'channel_id': video['snippet']['channelId'],
                'channel_title': video['snippet']['channelTitle'],
                'view_count': int(video['statistics'].get('viewCount', 0)),
                'like_count': int(video['statistics'].get('likeCount', 0)),
                'comment_count': int(video['statistics'].get('commentCount', 0)),
                'duration': video['contentDetails']['duration'],
                'thumbnail_url': video['snippet']['thumbnails']['maxres']['url']
                if 'maxres' in video['snippet']['thumbnails']
                else video['snippet']['thumbnails']['high']['url']
            }
            
        except HttpError as e:
            print(f"Error fetching details for video {video_id}: {e}")
            return None

    def process_thumbnails(self, videos: List[Dict]) -> Dict[str, str]:
        """Download and process thumbnails for collected videos.
        
        Args:
            videos: List of video data dictionaries
            
        Returns:
            Dictionary of errors if any occurred during download
        """
        return self.thumbnail_manager.batch_download(videos)

    def collect_data(self):
        """Main method to collect data from all target channels."""
        all_videos = []
        videos_per_channel = {}  # Track videos per channel
        
        # First pass: collect initial batch from each channel
        for channel_id in tqdm(TARGET_CHANNELS, desc="Processing channels - Initial batch"):
            videos = self.get_channel_videos(channel_id, VIDEOS_PER_CHANNEL // 2)
            if videos:
                channel_title = videos[0]['channel_title']
                videos_per_channel[channel_title] = len(videos)
                all_videos.extend(videos)
                
        # Second pass: collect remaining videos to balance channels
        min_videos = min(videos_per_channel.values()) if videos_per_channel else 0
        target_videos = max(min_videos, VIDEOS_PER_CHANNEL // 2)
        
        for channel_id in tqdm(TARGET_CHANNELS, desc="Processing channels - Balancing"):
            if not videos_per_channel:  # Skip if first pass failed
                continue
            channel_videos = self.get_channel_videos(channel_id, target_videos)
            if channel_videos:
                channel_title = channel_videos[0]['channel_title']
                current_count = videos_per_channel.get(channel_title, 0)
                remaining = target_videos - current_count
                if remaining > 0:
                    all_videos.extend(channel_videos[:remaining])
        
        # Process thumbnails
        print("\nDownloading thumbnails...")
        download_errors = self.process_thumbnails(all_videos)
        
        # Filter out videos with failed downloads
        successful_videos = [
            video for video in all_videos 
            if video['video_id'] not in download_errors
        ]
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(successful_videos, f, indent=2)
        
        # Create a summary DataFrame
        df = pd.DataFrame(successful_videos)
        summary_path = RAW_DATA_DIR / 'video_summary.csv'
        df.to_csv(summary_path, index=False)
        
        # Print collection summary
        print(f"\nCollection Summary:")
        print(f"Total videos collected: {len(all_videos)}")
        print(f"Successful downloads: {len(successful_videos)}")
        print(f"Failed downloads: {len(download_errors)}")
        if download_errors:
            print("\nDownload Errors:")
            for video_id, error in download_errors.items():
                print(f"- Video {video_id}: {error}")
        
        print(f"\nData saved to {self.metadata_path}")
        print(f"Summary saved to {summary_path}")
        
        # Cleanup any unused thumbnails
        self.thumbnail_manager.cleanup_unused([v['video_id'] for v in successful_videos])

if __name__ == "__main__":
    collector = YouTubeDataCollector()
    collector.collect_data()
