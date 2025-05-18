import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import custom preprocessing from your existing code
# Replace this import with the correct path to your preprocessing module
from preprocessing.preprocessing_from_db import custom_preprocessing

# Set constants
IMAGE_SIZE = (224, 224)

def load_model(pretrained=True):
    """
    Load ShuffleNetV2 model from torchvision
    
    Args:
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: The complete ShuffleNetV2 model
        feature_extractor: Model without the classifier layer
        device: Device the model is loaded on
    """
    # Set device (use CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ShuffleNetV2 model
    if pretrained:
        model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    else:
        model = shufflenet_v2_x1_0(weights=None)
        
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create feature extractor by removing classifier
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    return model, feature_extractor, device

def load_image(image_path, device):
    """
    Load and preprocess an image for ShuffleNetV2
    
    Args:
        image_path: Path to the image
        device: Device to load the image to
    
    Returns:
        Preprocessed image tensor
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open image using PIL
    img = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    img = transform(img)
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    # Move to device
    img = img.to(device)
    
    return img

def custom_preprocessing_wrapper(image):
    """
    A wrapper function for the imported custom_preprocessing function
    
    This is needed to adapt between different interfaces if necessary.
    The imported custom_preprocessing might expect and return different formats.
    
    Args:
        image: The input image (numpy array)
        
    Returns:
        Preprocessed image in PIL Image format
    """
    # Process the image using the imported function
    processed = custom_preprocessing(image)
    
    # Convert to PIL Image if necessary
    if isinstance(processed, np.ndarray):
        if processed.dtype != np.uint8:
            if processed.max() <= 1.0:
                processed = (processed * 255).astype(np.uint8)
            else:
                processed = processed.astype(np.uint8)
        return Image.fromarray(processed)
    
    return processed

def extract_features(image_path, use_custom_preprocessing=False, pretrained=True):
    """
    Extract features from an image using ShuffleNetV2
    
    Args:
        image_path: Path to the image
        use_custom_preprocessing: Whether to use custom preprocessing
        pretrained: Whether to use pretrained weights
    
    Returns:
        features: Feature maps extracted from the image (numpy array)
    """
    # Load model
    _, feature_extractor, device = load_model(pretrained)
    
    if use_custom_preprocessing:
        # Load image with OpenCV for custom preprocessing
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = cv2.resize(img_cv, IMAGE_SIZE)
        
        # Use imported custom preprocessing
        processed_img = custom_preprocessing_wrapper(img_cv)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = transform(processed_img)
        img = img.unsqueeze(0).to(device)
    else:
        # Use standard preprocessing
        img = load_image(image_path, device)
    
    # Extract features
    with torch.no_grad():  # Disable gradient computation for inference
        features = feature_extractor(img)
    
    # Move to CPU and convert to NumPy
    features = features.cpu().numpy()
    
    return features

def visualize_features(image_path, features, output_dir=None):
    """
    Visualize feature maps from ShuffleNetV2
    
    Args:
        image_path: Path to the original image
        features: Feature maps from the model
        output_dir: Directory to save visualizations (if None, won't save)
    """
    # Load original image for visualization
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, IMAGE_SIZE)
    
    # Load standard preprocessed image using PIL
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize(IMAGE_SIZE)
    preprocessed_img = np.array(img_pil)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Preprocessed image
    plt.subplot(1, 3, 2)
    plt.imshow(preprocessed_img)
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    # Average feature map
    plt.subplot(1, 3, 3)
    # Features shape: [batch, channels, height, width]
    avg_feature = np.mean(features[0], axis=0)
    plt.imshow(avg_feature, cmap='viridis')
    plt.title('Average Feature Map')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_features.png')
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    # Show individual feature channels
    num_channels = min(16, features.shape[1])
    plt.figure(figsize=(15, 15))
    
    for i in range(num_channels):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[0, i], cmap='viridis')
        plt.title(f'Channel {i}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization if output_dir is provided
    if output_dir:
        channels_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_channels.png')
        plt.savefig(channels_path)
    
    plt.show()

def get_feature_map(image_path, show_visualization=True, output_dir=None, use_custom_preprocessing=False):
    """
    Extract and optionally visualize features from an image
    
    Args:
        image_path: Path to the image
        show_visualization: Whether to show feature map visualization
        output_dir: Directory to save visualizations (if None, won't save)
        use_custom_preprocessing: Whether to use custom preprocessing pipeline
    
    Returns:
        features: Feature maps extracted from the image
    """
    # Extract features
    features = extract_features(image_path, use_custom_preprocessing)
    
    print(f"Feature map shape: {features.shape}")
    
    # Optionally visualize feature maps
    if show_visualization:
        visualize_features(image_path, features, output_dir)
    
    return features

def main():
    """
    Main function to demonstrate feature extraction
    """
    # Define an example image path - replace with your own image path
    image_path = r"C:\Users\dror\study\final_project\DataSets\processed_dataset\training\mandarins\obj0001\682999fcbde88bf1276b416e.png"
    
    # Extract features with visualization
    features = get_feature_map(
        image_path,
        show_visualization=True,
        use_custom_preprocessing=True  # Using the imported custom preprocessing
    )
    
    print("Feature extraction complete!")
    return features

if __name__ == "__main__":
    main()