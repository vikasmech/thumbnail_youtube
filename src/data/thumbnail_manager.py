"""Module for handling YouTube thumbnail downloads and management."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image
import io
from tqdm import tqdm
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import RAW_DATA_DIR, IMAGE_SIZE

class ThumbnailManager:
    """Handles downloading, processing, and managing YouTube thumbnails."""
    
    def __init__(self):
        """Initialize the ThumbnailManager."""
        self.thumbnails_dir = RAW_DATA_DIR / 'thumbnails'
        self.metadata_path = RAW_DATA_DIR / 'thumbnail_metadata.json'
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata if available
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load existing thumbnail metadata.
        
        Returns:
            Dictionary containing thumbnail metadata
        """
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_metadata(self):
        """Save thumbnail metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _validate_image(self, image_data: bytes) -> Tuple[bool, Optional[str]]:
        """Validate downloaded image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to open the image
            img = Image.open(io.BytesIO(image_data))
            
            # Check image mode (we want RGB or RGBA)
            if img.mode not in ['RGB', 'RGBA']:
                return False, f"Invalid image mode: {img.mode}"
                
            # Check minimum dimensions
            if img.size[0] < 100 or img.size[1] < 100:
                return False, f"Image too small: {img.size}"
                
            # Check file size (>1KB and <10MB)
            file_size = len(image_data)
            if file_size < 1024:
                return False, f"File too small: {file_size} bytes"
            if file_size > 10 * 1024 * 1024:
                return False, f"File too large: {file_size} bytes"
                
            return True, None
            
        except Exception as e:
            return False, str(e)
            
    def _compute_image_hash(self, image_data: bytes) -> str:
        """Compute a hash of the image data for deduplication.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            SHA-256 hash of the image data
        """
        return hashlib.sha256(image_data).hexdigest()
        
    def download_thumbnail(self, video_id: str, url: str, 
                         max_retries: int = 3) -> Tuple[bool, Optional[str]]:
        """Download and save a video thumbnail.
        
        Args:
            video_id: The YouTube video ID
            url: URL of the thumbnail image
            max_retries: Maximum number of download attempts
            
        Returns:
            Tuple of (success, error_message)
        """
        # Check if already downloaded
        if video_id in self.metadata:
            return True, None
            
        for attempt in range(max_retries):
            try:
                # Download image
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image_data = response.content
                
                # Validate image
                is_valid, error = self._validate_image(image_data)
                if not is_valid:
                    if attempt == max_retries - 1:
                        return False, f"Invalid image: {error}"
                    continue
                
                # Check for duplicates
                image_hash = self._compute_image_hash(image_data)
                
                # Save image
                thumbnail_path = self.thumbnails_dir / f"{video_id}.jpg"
                with open(thumbnail_path, 'wb') as f:
                    f.write(image_data)
                
                # Update metadata
                self.metadata[video_id] = {
                    'url': url,
                    'file_path': str(thumbnail_path),
                    'hash': image_hash,
                    'downloaded_at': str(Path(thumbnail_path).stat().st_mtime)
                }
                self._save_metadata()
                
                return True, None
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    return False, f"Download failed: {str(e)}"
                continue
                
            except Exception as e:
                return False, f"Unexpected error: {str(e)}"
                
    def batch_download(self, video_data: List[Dict], 
                      max_workers: int = 5) -> Dict[str, str]:
        """Download multiple thumbnails in parallel.
        
        Args:
            video_data: List of video data dictionaries containing 'video_id' and 'thumbnail_url'
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping video IDs to error messages (empty if successful)
        """
        errors = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create download tasks
            future_to_video = {
                executor.submit(
                    self.download_thumbnail, 
                    video['video_id'], 
                    video['thumbnail_url']
                ): video['video_id']
                for video in video_data
                if video['video_id'] not in self.metadata  # Skip already downloaded
            }
            
            # Process results with progress bar
            with tqdm(total=len(future_to_video), 
                     desc="Downloading thumbnails") as pbar:
                for future in as_completed(future_to_video):
                    video_id = future_to_video[future]
                    try:
                        success, error = future.result()
                        if not success:
                            errors[video_id] = error
                    except Exception as e:
                        errors[video_id] = str(e)
                    pbar.update(1)
        
        return errors
        
    def cleanup_unused(self, valid_video_ids: List[str]):
        """Remove thumbnails that are no longer needed.
        
        Args:
            valid_video_ids: List of video IDs that should be kept
        """
        # Find thumbnails to remove
        to_remove = set(self.metadata.keys()) - set(valid_video_ids)
        
        # Remove files and metadata
        for video_id in to_remove:
            try:
                # Remove file
                thumbnail_path = Path(self.metadata[video_id]['file_path'])
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                
                # Remove metadata
                del self.metadata[video_id]
                
            except Exception as e:
                print(f"Error removing thumbnail {video_id}: {e}")
        
        # Save updated metadata
        self._save_metadata()
