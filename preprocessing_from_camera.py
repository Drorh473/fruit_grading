import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from camera_simulator import CameraSimulator
from tqdm import tqdm

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

def get_processed_images(db_name="fruit_grading", collection_name="images", fps=300):
    """
    Get images from the camera simulator and apply preprocessing
    
    Args:
        db_name: MongoDB database name
        collection_name: MongoDB collection name
        num_images: Number of images to process
        fps: Frames per second for the camera simulator
        
    Returns:
        List of processed image dictionaries with metadata
    """
    # Create and start camera simulator
    camera = CameraSimulator(
        db_name=db_name,
        collection_name=collection_name,
        fps=fps,
        random_order=True
    )
    
    camera.start()
    
    # Process images
    processed_images = []
    
    try:
        for i in tqdm(range(len(camera.doc_ids)),desc="proccesing images from camera"):
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

def display_processed_images(processed_images):
    """
    Display original and processed images side by side
    
    Args:
        processed_images: List of processed image dictionaries
    """
    num_images = len(processed_images)
    if num_images == 0:
        print("No images to display")
        return
    
    # Create figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    # Handle case of single image
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Display each image
    for i, processed in enumerate(processed_images):
        # Get metadata
        metadata = processed.get('metadata', {})
        category = metadata.get('category', 'unknown')
        
        # Display original
        axes[i, 0].imshow(processed['original'])
        axes[i, 0].set_title(f"Original - {category}")
        axes[i, 0].axis('off')
        
        # Display after Gaussian blur
        axes[i, 1].imshow(processed['blurred'])
        axes[i, 1].set_title("Gaussian Blur")
        axes[i, 1].axis('off')
        
        # Display after CLAHE
        axes[i, 2].imshow(processed['enhanced'])
        axes[i, 2].set_title("CLAHE Enhanced")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("preprocessing_results.png")
    plt.show()

def main():
    """Main function to demonstrate preprocessing on camera images"""
    print("Getting images from camera simulator and applying preprocessing...")
    
    # Process 4 images
    processed_images = get_processed_images(
        db_name="fruit_grading", 
        collection_name="images",
        fps=300
    )
    
    # Display results
    if processed_images:
        print(f"Displaying {len(processed_images)} processed images")
        display_processed_images(processed_images[:5])
    else:
        print("No images were processed successfully")

if __name__ == "__main__":
    main()