"""Quick utility to check image properties of downloaded thumbnails."""
import os
from pathlib import Path
import random
from PIL import Image
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR

def check_random_thumbnails(num_samples=5):
    """Check properties of random thumbnail samples."""
    thumbnails_dir = RAW_DATA_DIR / 'thumbnails'
    all_thumbnails = list(thumbnails_dir.glob('*.jpg'))
    
    if not all_thumbnails:
        print("No thumbnails found!")
        return
        
    samples = random.sample(all_thumbnails, min(num_samples, len(all_thumbnails)))
    
    print(f"\nChecking {len(samples)} random thumbnails:")
    print("=" * 50)
    
    for thumb_path in samples:
        print(f"\nThumbnail: {thumb_path.name}")
        img = Image.open(thumb_path)
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size}")
        print(f"File size: {thumb_path.stat().st_size / 1024:.1f} KB")
        
if __name__ == "__main__":
    check_random_thumbnails()
