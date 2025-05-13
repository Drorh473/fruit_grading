import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dotenv import load_dotenv
from camera_simulator import CameraSimulator
from tqdm import tqdm

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Get MongoDB settings
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')

# Get camera settings
CAMERA_FPS = int(os.getenv('CAMERA_FPS', 30))
CAMERA_RANDOM_ORDER = os.getenv('CAMERA_RANDOM_ORDER', 'true').lower() == 'true'

def custom_preprocessing(image):
    """Apply preprocessing steps from the article:
    1. Gaussian filtering for noise removal
    2. CLAHE for histogram equalization
    """
    # Make sure image is in uint8 format for OpenCV
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    h, w = image.shape[:2]
    max_dim = 1000  # Maximum dimension size
    if h > max_dim or w > max_dim:
        # Calculate the ratio to resize
        ratio = max_dim / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))
    # Save original image for comparison
    original = image.copy()
    
    # 1. Apply Gaussian filter with kernel size 3x3 and sigma=0.01
    blurred = cv2.GaussianBlur(image, (3, 3), 0.001)
    
    # 2. Apply CLAHE for histogram equalization (on grayscale version)
    # Convert to Lab color space (L channel is the lightness)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1] range for the neural network
    enhanced_normalized = enhanced.astype(np.float32) / 255.0
    
    # Return both the processed images and the normalized version
    return {
        'original': original,
        'blurred': blurred, 
        'enhanced': enhanced,
        'normalized': enhanced_normalized
    }

def get_processed_images(db_name=None, collection_name="images", fps=None, num_images=None):
    """
    Get images from the camera simulator and apply preprocessing
    
    Args:
        db_name: MongoDB database name (defaults to env setting)
        collection_name: MongoDB collection name
        fps: Frames per second for the camera simulator (defaults to env setting)
        num_images: Number of images to process (if None, processes all images)
        
    Returns:
        List of processed image dictionaries with metadata
    """
    # Use environment settings if not specified
    db_name = db_name or DB_NAME
    fps = fps or CAMERA_FPS
    
    # Create and start camera simulator
    camera = CameraSimulator(
        db_name=db_name,
        collection_name=collection_name,
        fps=fps,
        random_order=CAMERA_RANDOM_ORDER
    )
    
    camera.start()
    
    # Process images
    processed_images = []
    
    try:
        # Determine how many images to process
        if num_images is None:
            num_images = len(camera.doc_ids)
        else:
            num_images = min(num_images, len(camera.doc_ids))
        
        for i in tqdm(range(num_images), desc="Processing images from camera"):
            # Read image from camera
            success, frame, metadata = camera.read()
            
            if not success:
                print(f"Failed to read image {i+1}")
                continue
            
            # Apply preprocessing
            processed = custom_preprocessing(frame)
            
            # Add metadata
            processed['metadata'] = metadata
            
            # Add to list
            processed_images.append(processed)
    
    finally:
        # Stop the camera
        camera.stop()
    
    return processed_images