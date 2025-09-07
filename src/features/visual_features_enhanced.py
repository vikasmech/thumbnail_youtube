"""Enhanced module for extracting visual features from thumbnails."""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
import colorsys
from scipy.stats import entropy
from scipy import ndimage

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, FACE_DETECTION_CONFIDENCE

# Load OpenCV models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class EnhancedThumbnailFeatureExtractor:
    """Enhanced feature extractor with additional metrics."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.thumbnails_dir = RAW_DATA_DIR / 'thumbnails'
        self.features_dir = PROCESSED_DATA_DIR / 'visual_features'
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_advanced_color_features(self, image: np.ndarray) -> Dict:
        """Extract enhanced color-related features.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing color features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract dominant colors with confidence scores
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, n_init=1).fit(pixels)
        colors = kmeans.cluster_centers_
        distances = kmeans.transform(pixels)  # Distance to each cluster center
        confidence_scores = 1 / (1 + distances.min(axis=1))  # Convert distances to confidence scores
        
        # Calculate color statistics per channel
        channel_stats = {
            'mean': np.mean(image, axis=(0,1)).tolist(),
            'std': np.std(image, axis=(0,1)).tolist(),
            'median': np.median(image, axis=(0,1)).tolist()
        }
        
        # Calculate color harmony
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()
        complementary_harmony = self._calculate_complementary_harmony(hue_hist)
        
        # Calculate color warmth
        warmth_score = self._calculate_color_warmth(image)
        
        # Calculate local contrast
        local_contrast = self._calculate_local_contrast(lab[:,:,0])
        
        # Calculate color zones
        height, width = image.shape[:2]
        top_colors = np.mean(image[:height//3], axis=(0,1)).tolist()
        middle_colors = np.mean(image[height//3:2*height//3], axis=(0,1)).tolist()
        bottom_colors = np.mean(image[2*height//3:], axis=(0,1)).tolist()
        
        return {
            'dominant_colors': colors.tolist(),
            'color_confidence': float(np.mean(confidence_scores)),
            'channel_stats': channel_stats,
            'complementary_harmony': float(complementary_harmony),
            'warmth_score': float(warmth_score),
            'local_contrast': float(local_contrast),
            'color_zones': {
                'top': top_colors,
                'middle': middle_colors,
                'bottom': bottom_colors
            },
            'brightness': float(np.mean(lab[:,:,0])),
            'saturation': float(np.mean(hsv[:,:,1])),
            'color_entropy': float(entropy(hue_hist[hue_hist > 0]))
        }
        
    def _calculate_complementary_harmony(self, hue_hist: np.ndarray) -> float:
        """Calculate color harmony based on complementary colors.
        
        Args:
            hue_hist: Histogram of hue values
            
        Returns:
            Harmony score between 0 and 1
        """
        # Shift histogram by 90 degrees (complementary colors)
        shifted_hist = np.roll(hue_hist, 90)
        # Calculate correlation between original and shifted
        correlation = np.corrcoef(hue_hist, shifted_hist)[0,1]
        # Convert to harmony score (0-1)
        return (correlation + 1) / 2
        
    def _calculate_color_warmth(self, image: np.ndarray) -> float:
        """Calculate color warmth score.
        
        Args:
            image: BGR image array
            
        Returns:
            Warmth score between 0 and 1
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Warm colors have more red and yellow (R and G channels)
        warmth = np.mean(rgb[:,:,0]) + np.mean(rgb[:,:,1]) - np.mean(rgb[:,:,2])
        # Normalize to 0-1
        return (warmth + 510) / 1020  # Max possible range is -510 to 510
        
    def _calculate_local_contrast(self, luminance: np.ndarray) -> float:
        """Calculate local contrast using Gaussian difference.
        
        Args:
            luminance: Luminance channel array
            
        Returns:
            Local contrast score
        """
        # Apply Gaussian blur at two scales
        blur1 = cv2.GaussianBlur(luminance, (3,3), 0)
        blur2 = cv2.GaussianBlur(luminance, (21,21), 0)
        # Calculate difference
        contrast = np.mean(np.abs(blur1 - blur2))
        return contrast / 255  # Normalize to 0-1
        
    def extract_enhanced_face_features(self, image: np.ndarray) -> Dict:
        """Extract enhanced face-related features.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing face features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with different scales
        faces_rect = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        total_face_area = 0
        face_brightness = []
        face_contrast = []
        
        for (x, y, w, h) in faces_rect:
            face_roi = gray[y:y+h, x:x+w]
            face_area = w * h
            total_face_area += face_area
            
            # Detect eyes in face region
            eyes = eye_cascade.detectMultiScale(face_roi)
            
            # Calculate face-specific metrics
            face_bright = np.mean(face_roi)
            face_brightness.append(face_bright)
            face_cont = np.std(face_roi)
            face_contrast.append(face_cont)
            
            # Calculate face position relative to rule of thirds points
            thirds_points = [
                (image.shape[1]/3, image.shape[0]/3),
                (2*image.shape[1]/3, image.shape[0]/3),
                (image.shape[1]/3, 2*image.shape[0]/3),
                (2*image.shape[1]/3, 2*image.shape[0]/3)
            ]
            
            face_center = (x + w/2, y + h/2)
            min_thirds_dist = min(
                np.sqrt((px - face_center[0])**2 + (py - face_center[1])**2)
                for px, py in thirds_points
            )
            
            face_info = {
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'relative_size': face_area / (image.shape[0] * image.shape[1]),
                'center': [float(x + w/2) / image.shape[1], 
                          float(y + h/2) / image.shape[0]],
                'aspect_ratio': w / h,
                'num_eyes': len(eyes),
                'brightness': float(face_bright),
                'contrast': float(face_cont),
                'thirds_distance': float(min_thirds_dist)
            }
            faces.append(face_info)
        
        # Calculate face clustering
        face_clustering = 0
        if len(faces) > 1:
            centers = np.array([[f['center'][0], f['center'][1]] for f in faces])
            face_clustering = np.mean([
                np.min([
                    np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    for j, c2 in enumerate(centers) if i != j
                ])
                for i, c1 in enumerate(centers)
            ])
        
        return {
            'num_faces': len(faces),
            'faces': faces,
            'has_faces': len(faces) > 0,
            'total_face_area_ratio': total_face_area / (image.shape[0] * image.shape[1]),
            'face_clustering': float(face_clustering),
            'avg_face_brightness': float(np.mean(face_brightness)) if faces else 0,
            'avg_face_contrast': float(np.mean(face_contrast)) if faces else 0
        }
        
    def analyze_enhanced_composition(self, image: np.ndarray) -> Dict:
        """Enhanced analysis of image composition.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary containing composition features
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (height * width)
        
        # Calculate edge direction histogram
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_directions = np.arctan2(sobely, sobelx)
        direction_hist, _ = np.histogram(edge_directions[edges > 0], bins=8, range=(-np.pi, np.pi))
        direction_hist = direction_hist / np.sum(direction_hist)
        
        # Calculate visual complexity using DCT
        dct = cv2.dct(np.float32(gray))
        visual_complexity = np.sum(np.abs(dct)) / (height * width)
        
        # Calculate rule of thirds adherence
        thirds_map = np.zeros((height, width))
        for i in range(1, 3):
            thirds_map[:, int(width * i/3)] = 1
            thirds_map[int(height * i/3), :] = 1
        thirds_score = np.sum(edges * cv2.GaussianBlur(thirds_map, (21,21), 0)) / np.sum(edges)
        
        # Calculate balance metrics
        left_weight = np.sum(gray[:, :width//2])
        right_weight = np.sum(gray[:, width//2:])
        top_weight = np.sum(gray[:height//2, :])
        bottom_weight = np.sum(gray[height//2:, :])
        
        horizontal_balance = 1 - abs(left_weight - right_weight) / (left_weight + right_weight)
        vertical_balance = 1 - abs(top_weight - bottom_weight) / (top_weight + bottom_weight)
        
        # Calculate focal points using alternative method
        # Convert to grayscale and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate gradient magnitude as a simple saliency map
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-1
        saliency_map = cv2.normalize(gradient_mag, None, 0, 1, cv2.NORM_MINMAX)
        focal_points = self._find_focal_points(saliency_map)
        
        return {
            'edge_density': float(edge_density),
            'edge_directions': direction_hist.tolist(),
            'visual_complexity': float(visual_complexity),
            'thirds_adherence': float(thirds_score),
            'balance': {
                'horizontal': float(horizontal_balance),
                'vertical': float(vertical_balance)
            },
            'focal_points': focal_points,
            'symmetry': {
                'horizontal': float(self._calculate_symmetry(gray, 'horizontal')),
                'vertical': float(self._calculate_symmetry(gray, 'vertical')),
                'radial': float(self._calculate_radial_symmetry(gray))
            }
        }
        
    def _find_focal_points(self, saliency_map: np.ndarray, 
                          threshold: float = 0.8) -> List[Dict]:
        """Find focal points in the image using saliency.
        
        Args:
            saliency_map: Saliency map array
            threshold: Saliency threshold
            
        Returns:
            List of focal point dictionaries
        """
        # Normalize saliency map
        saliency_norm = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Find local maxima
        local_max = ndimage.maximum_filter(saliency_norm, size=20)
        maxima = (saliency_norm == local_max) & (saliency_norm > threshold)
        
        focal_points = []
        coords = np.column_stack(np.where(maxima))
        
        for y, x in coords:
            focal_points.append({
                'position': [float(x) / saliency_map.shape[1],
                           float(y) / saliency_map.shape[0]],
                'saliency': float(saliency_norm[y, x])
            })
            
        return focal_points
        
    def _calculate_symmetry(self, image: np.ndarray, axis: str) -> float:
        """Calculate image symmetry along specified axis.
        
        Args:
            image: Grayscale image array
            axis: 'horizontal' or 'vertical'
            
        Returns:
            Symmetry score between 0 and 1
        """
        if axis == 'horizontal':
            left = image[:, :image.shape[1]//2]
            right = np.fliplr(image[:, image.shape[1]//2:])
            diff = np.abs(left - right)
        else:
            top = image[:image.shape[0]//2, :]
            bottom = np.flipud(image[image.shape[0]//2:, :])
            diff = np.abs(top - bottom)
            
        return 1 - np.mean(diff) / 255
        
    def _calculate_radial_symmetry(self, image: np.ndarray) -> float:
        """Calculate radial symmetry of the image.
        
        Args:
            image: Grayscale image array
            
        Returns:
            Radial symmetry score between 0 and 1
        """
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        
        # Calculate distances from center
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.max(distances)
        
        # Create rings
        num_rings = 10
        ring_width = max_distance / num_rings
        
        symmetry_scores = []
        for i in range(num_rings):
            ring_mask = (distances >= i*ring_width) & (distances < (i+1)*ring_width)
            ring_pixels = image[ring_mask]
            if len(ring_pixels) > 0:
                ring_std = np.std(ring_pixels)
                symmetry_scores.append(1 - ring_std / 255)
                
        return np.mean(symmetry_scores)
        
    def extract_features(self, video_id: str) -> Optional[Dict]:
        """Extract all enhanced features for a single thumbnail.
        
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
                'color_features': self.extract_advanced_color_features(image),
                'face_features': self.extract_enhanced_face_features(image),
                'composition_features': self.analyze_enhanced_composition(image)
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
        
        for thumb_path in tqdm(thumbnail_files, desc="Extracting enhanced features"):
            video_id = thumb_path.stem
            features = self.extract_features(video_id)
            
            if features:
                all_features.append(features)
            else:
                failed.append(video_id)
                
        # Save features
        features_path = self.features_dir / 'enhanced_thumbnail_features.json'
        with open(features_path, 'w') as f:
            json.dump(all_features, f, indent=2)
            
        # Print summary
        print(f"\nEnhanced Feature Extraction Summary:")
        print(f"Successfully processed: {len(all_features)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed videos:")
            for video_id in failed:
                print(f"- {video_id}")
        
        print(f"\nFeatures saved to {features_path}")

if __name__ == "__main__":
    extractor = EnhancedThumbnailFeatureExtractor()
    extractor.process_all_thumbnails()
