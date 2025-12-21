import numpy as np
import cv2
from pathlib import Path


def create_test_image(size=(224, 224, 3), color='random'):
    """Create a test image
    
    Args:
        size: Image dimensions (height, width, channels)
        color: 'random', 'red', 'green', 'blue', 'black', 'white'
    
    Returns:
        numpy array representing image
    """
    if color == 'random':
        return np.random.randint(0, 255, size, dtype=np.uint8)
    elif color == 'red':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel
        return img
    elif color == 'green':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 1] = 255  # Green channel
        return img
    elif color == 'blue':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel
        return img
    elif color == 'black':
        return np.zeros(size, dtype=np.uint8)
    elif color == 'white':
        return np.ones(size, dtype=np.uint8) * 255


def save_test_image(path, size=(224, 224, 3), color='random'):
    """Create and save a test image
    
    Args:
        path: Path to save image
        size: Image dimensions
        color: Image color
    """
    img = create_test_image(size, color)
    cv2.imwrite(str(path), img)
    return path


def create_test_dataset(output_dir, num_images=10, num_cameras=4):
    """Create a complete test dataset
    
    Args:
        output_dir: Directory to save images
        num_images: Number of fruit objects
        num_cameras: Number of camera angles
    
    Returns:
        List of image metadata dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    fruit_types = ['market', 'standard', 'premium']
    
    for i in range(num_images):
        fruit_type = fruit_types[i % len(fruit_types)]
        object_id = f"obj{i:04d}"
        
        for camera_id in range(num_cameras):
            # Create directory structure
            camera_dir = output_dir / fruit_type / f"camera_{camera_id}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Create image
            img_name = f"{object_id}_cam{camera_id}.png"
            img_path = camera_dir / img_name
            
            save_test_image(img_path)
            
            # Create metadata
            metadata = {
                "path": str(img_path),
                "fruit_type": fruit_type,
                "object_id": object_id,
                "camera_id": camera_id,
                "timestamp": f"2025-01-01T00:{i:02d}:{camera_id:02d}",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": "A"
            }
            
            metadata_list.append(metadata)
    
    return metadata_list


def compare_images(img1, img2, tolerance=1e-5):
    """Compare two images for similarity
    
    Args:
        img1: First image
        img2: Second image
        tolerance: Maximum allowed difference
    
    Returns:
        Boolean indicating if images are similar
    """
    if img1.shape != img2.shape:
        return False
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = np.max(diff)
    
    return max_diff < tolerance


def assert_valid_image_metadata(metadata):
    """Assert that image metadata has all required fields
    
    Args:
        metadata: Dictionary of image metadata
    """
    required_fields = [
        'path', 'fruit_type', 'object_id', 'camera_id',
        'timestamp', 'width', 'height'
    ]
    
    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"
