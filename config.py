"""Configuration settings for the YouTube thumbnail analysis project."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Directory paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = ROOT_DIR / 'reports'

# Data collection settings
TARGET_CHANNELS = [
    # Tech Reviews & News
    'UCXuqSBlHAE6Xw-yeJA0Tunw',  # Linus Tech Tips (39.14 uploads/month)
    'UC2UXDak6o7rBm23k3Vv5dww',  # Techno Ruhez (7.42 uploads/month)
    'UCBJycsmduvYEL83R_U4JriQ',  # MKBHD
    
    # Tech Education
    'UC8butISFwT-Wl7EV0hUK0BQ',  # freeCodeCamp (17.39 uploads/month)
    'UCeVMnSShP_Iviwkknt83cww',  # Code with Harry (7.61 uploads/month)
    'UCFbNIlppjAuEX4znoulh0Cw',  # Web Dev Simplified (6.21 uploads/month)
    
    # Business & Tech
    'UCsBjURrPoezykLs9EqgamOA',  # Fireship
    'UCZ9qFEC82qM6Pk-54Q4TVWA',  # FastAI
    'UCxX9wt5FWQUAAz4UrysqK9A',  # CS Dojo
    'UC4xKdmAXFh4ACyhpiQ_3qBw'   # TechLead
]
VIDEOS_PER_CHANNEL = 200  # Target 200 videos per channel to exceed 500+ total

# Feature extraction settings
IMAGE_SIZE = (224, 224)  # Standard size for thumbnail processing
FACE_DETECTION_CONFIDENCE = 0.9
COLOR_PALETTE_SIZE = 5  # Number of dominant colors to extract

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
