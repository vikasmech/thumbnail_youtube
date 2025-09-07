"""Module for extracting visual features from thumbnails."""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
import pytesseract
from sklearn.cluster import KMeans
import json
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, FACE_DETECTION_CONFIDENCE

class ThumbnailFeatureExtractor:
    """Extracts visual features from YouTube thumbnails."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.thumbnails_dir = RAW_DATA_DIR / 'thumbnails'
        self.features_dir = PROCESSED_DATA_DIR / 'visual_features'
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_color_features(self, image: np.ndarray) -> Dict:
        """Extract color-related features from the image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing color features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract dominant colors using k-means
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, n_init=1).fit(pixels)
        colors = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)
        percentages = counts / len(pixels)
        
        # Calculate color statistics
        brightness = np.mean(lab[:,:,0])
        saturation = np.mean(hsv[:,:,1])
        
        # Calculate color contrast
        lab_std = np.std(lab[:,:,0])
        contrast = lab_std / 100.0  # Normalize to 0-1
        
        # Calculate color diversity (entropy of color distribution)
        hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        non_zero = hist[hist > 0]
        color_entropy = -np.sum(non_zero * np.log2(non_zero))
        
        return {
            'dominant_colors': colors.tolist(),
            'color_percentages': percentages.tolist(),
            'brightness': float(brightness),
            'saturation': float(saturation),
            'contrast': float(contrast),
            'color_entropy': float(color_entropy)
        }
        
    def extract_face_features(self, image: np.ndarray) -> Dict:
        """Extract face-related features from the image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing face features
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_rect = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in faces_rect:
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            
            face_info = {
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'relative_size': face_area / image_area,
                'center': [(x + w/2) / image.shape[1], 
                          (y + h/2) / image.shape[0]]
            }
            faces.append(face_info)
            
        return {
            'num_faces': len(faces),
            'faces': faces,
            'has_faces': len(faces) > 0
        }
        
    def extract_text_features(self, image: np.ndarray) -> Dict:
        """Extract text-related features from the image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing text features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text using Tesseract
        text = pytesseract.image_to_string(gray)
        words = [w for w in text.split() if len(w) > 1]  # Filter single chars
        
        # Get text bounding boxes
        boxes = pytesseract.image_to_boxes(gray)
        text_regions = []
        
        if boxes:
            for box in boxes.splitlines():
                b = box.split()
                if len(b) == 6:  # Standard box format
                    x1, y1, x2, y2 = map(int, b[1:5])
                    # Convert to relative coordinates
                    text_regions.append({
                        'bbox': [
                            x1 / image.shape[1],
                            y1 / image.shape[0],
                            x2 / image.shape[1],
                            y2 / image.shape[0]
                        ]
                    })
        
        return {
            'has_text': len(words) > 0,
            'word_count': len(words),
            'text_content': text.strip(),
            'text_regions': text_regions
        }
        
    def analyze_composition(self, image: np.ndarray) -> Dict:
        """Analyze image composition features.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing composition features
        """
        height, width = image.shape[:2]
        
        # Calculate edge density using Canny
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (height * width)
        
        # Calculate rule of thirds points
        third_h = height // 3
        third_w = width // 3
        thirds_points = [
            (third_w, third_h),
            (third_w * 2, third_h),
            (third_w, third_h * 2),
            (third_w * 2, third_h * 2)
        ]
        
        # Calculate visual weight distribution
        gray = gray.astype(float) / 255
        weight_left = np.mean(gray[:, :width//2])
        weight_right = np.mean(gray[:, width//2:])
        weight_top = np.mean(gray[:height//2, :])
        weight_bottom = np.mean(gray[height//2:, :])
        
        # Calculate symmetry
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        horizontal_symmetry = 1 - np.mean(np.abs(left_half - right_half))
        
        top_half = gray[:height//2, :]
        bottom_half = np.flipud(gray[height//2:, :])
        vertical_symmetry = 1 - np.mean(np.abs(top_half - bottom_half))
        
        return {
            'edge_density': float(edge_density),
            'weight_distribution': {
                'left': float(weight_left),
                'right': float(weight_right),
                'top': float(weight_top),
                'bottom': float(weight_bottom)
            },
            'symmetry': {
                'horizontal': float(horizontal_symmetry),
                'vertical': float(vertical_symmetry)
            },
            'thirds_points': thirds_points
        }
        
    def extract_features(self, video_id: str) -> Optional[Dict]:
        """Extract all features for a single thumbnail.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing all extracted features or None if error
        """
        try:
            # Load image
            image_path = self.thumbnails_dir / f"{video_id}.jpg"
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Failed to load image for video {video_id}")
                return None
                
            # Resize for consistent processing
            image = cv2.resize(image, IMAGE_SIZE)
            
            # Extract all features
            features = {
                'video_id': video_id,
                'color_features': self.extract_color_features(image),
                'face_features': self.extract_face_features(image),
                'text_features': self.extract_text_features(image),
                'composition_features': self.analyze_composition(image)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for video {video_id}: {e}")
            return None
            
    def process_all_thumbnails(self):
        """Extract features from all thumbnails in the dataset."""
        # Get list of all thumbnails
        thumbnail_files = list(self.thumbnails_dir.glob('*.jpg'))
        
        if not thumbnail_files:
            print("No thumbnails found!")
            return
            
        # Process all thumbnails
        all_features = []
        failed = []
        
        for thumb_path in tqdm(thumbnail_files, desc="Extracting features"):
            video_id = thumb_path.stem
            features = self.extract_features(video_id)
            
            if features:
                all_features.append(features)
            else:
                failed.append(video_id)
                
        # Save features
        features_path = self.features_dir / 'thumbnail_features.json'
        with open(features_path, 'w') as f:
            json.dump(all_features, f, indent=2)
            
        # Print summary
        print(f"\nFeature Extraction Summary:")
        print(f"Successfully processed: {len(all_features)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed videos:")
            for video_id in failed:
                print(f"- {video_id}")
        
        print(f"\nFeatures saved to {features_path}")

if __name__ == "__main__":
    extractor = ThumbnailFeatureExtractor()
    extractor.process_all_thumbnails()
