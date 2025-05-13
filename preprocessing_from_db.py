import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Get dataset paths from environment
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')

def custom_preprocessing(image, save_path=None):
    """Apply preprocessing steps from the article:
    1. Gaussian filtering for noise removal
    2. CLAHE for histogram equalization
    
    Args:
        image: Input image to process
        save_path: If provided, save the processed image to this path
    """
    # Make sure image is in uint8 format for OpenCV
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    max_dim = 224  # Maximum dimension size
    if h > max_dim or w > max_dim:
        # Calculate the ratio to resize
        ratio = max_dim / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))
    
    # Save original image for comparison
    original = image.copy()
    
    # 1. Apply Gaussian filter with kernel size 3x3 and sigma=0.01
    blurred = cv2.GaussianBlur(original, (3, 3), 0.001)
    
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
    
    # Save the processed image if a path is provided
    if save_path:
        # Convert to BGR for OpenCV saving
        save_img = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, save_img)
    
    # Normalize to [0,1] range for the neural network
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced

def preprocess_and_save_dataset(input_dir=None, output_dir=None):
    """Preprocess all images in the dataset and save them to the output directory"""
    # Use environment paths if not provided
    input_dir = input_dir or ORIGINAL_DATASET_PATH
    output_dir = output_dir or PROCESSED_DATASET_PATH
    
    print(f"Starting preprocessing and saving images from {input_dir} to {output_dir}...")
    
    # Create output directories
    for dataset_type in ['training', 'testing', 'validation']:
        input_type_dir = os.path.join(input_dir, dataset_type)
        if not os.path.exists(input_type_dir):
            continue
            
        # Get all class directories
        class_dirs = [d for d in os.listdir(input_type_dir) if os.path.isdir(os.path.join(input_type_dir, d))]
        
        for class_name in class_dirs:
            # Create output class directory
            output_class_dir = os.path.join(output_dir, dataset_type, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images in this class
            class_dir = os.path.join(input_type_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {len(images)} images in {dataset_type}/{class_name}")
            
            # Process each image
            for img_name in images:
                input_path = os.path.join(class_dir, img_name)
                output_path = os.path.join(output_class_dir, img_name)
                
                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Warning: Could not read {input_path}")
                    continue
                
                # Convert to RGB for processing
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing and save
                custom_preprocessing(img, save_path=output_path)
    
    print(f"Preprocessing complete. All images saved to {output_dir}")
    return output_dir

class SimpleImageGenerator:
    def __init__(self, preprocessing_function=None, shuffle=True):
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
    
    def flow_from_directory(self, directory, target_size, batch_size, class_mode='categorical', shuffle=True):
        # Create a dataset loader that reads from directory
        return DirectoryIterator(
            directory,
            self.preprocessing_function,
            target_size,
            batch_size,
            class_mode,
            shuffle
        )

class DirectoryIterator:
    def __init__(self, directory, preprocessing_function, target_size, batch_size, class_mode='categorical', shuffle=True):
        self.directory = directory
        self.preprocessing_function = preprocessing_function
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.shuffle = shuffle
        
        # Get class directories
        self.class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_dirs)}
        
        # Get all image paths and labels
        self.filenames = []
        self.classes = []
        
        for class_name in self.class_dirs:
            class_dir = os.path.join(directory, class_name)
            class_idx = self.class_indices[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.filenames.append(os.path.join(class_dir, img_name))
                    self.classes.append(class_idx)
        
        self.samples = len(self.filenames)
        self.n = self.samples
        self.indexes = np.arange(self.samples)
        self.current_index = 0
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.samples:
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration
        
        # Get current batch indexes
        batch_indexes = self.indexes[self.current_index:min(self.current_index + self.batch_size, self.samples)]
        self.current_index += self.batch_size
        
        # Prepare batch data
        batch_x = np.zeros((len(batch_indexes), self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_indexes), len(self.class_indices)), dtype=np.float32)
        
        # Load and preprocess images
        for i, idx in enumerate(batch_indexes):
            # Load image
            img = cv2.imread(self.filenames[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            
            # Apply preprocessing if available
            if self.preprocessing_function:
                img = self.preprocessing_function(img)
            
            batch_x[i] = img
            
            # One-hot encode class
            batch_y[i, self.classes[idx]] = 1.0
        
        return batch_x, batch_y
    
    def __len__(self):
        return math.ceil(self.samples / self.batch_size)

    def take(self, count):
        """Return the specified number of batches"""
        batches = []
        for _ in range(count):
            try:
                batch = next(self)
                batches.append(batch)
            except StopIteration:
                break
        return batches

def load_dataset_with_preprocessing(base_dir=None, img_size=(224, 224), batch_size=32, preprocess_first=False):
    """
    Load dataset with custom preprocessing
    
    Args:
        base_dir: Base directory containing training/testing/validation folders
        img_size: Target image size
        batch_size: Batch size for the data generators
        preprocess_first: If True, preprocess all images first and save them
    """
    # Use environment paths if not provided
    if base_dir is None:
        base_dir = PROCESSED_DATASET_PATH
    
    print(f"Loading dataset from {base_dir} with custom preprocessing...")
    
    # Option to preprocess and save all images first
    if preprocess_first:
        # Use original dataset path from environment
        original_path = ORIGINAL_DATASET_PATH
        if not original_path:
            print("Warning: ORIGINAL_DATASET_PATH not set in .env file. Creating a backup.")
            if os.path.exists(base_dir) and not os.path.exists(f"{base_dir}_original"):
                print(f"Creating backup of original images in {base_dir}_original")
                shutil.copytree(base_dir, f"{base_dir}_original")
                original_path = f"{base_dir}_original"
        
        # Preprocess and save all images
        preprocess_and_save_dataset(original_path, base_dir)
        # Since images are already preprocessed, we won't apply preprocessing again
        preprocessing_fn = None
    else:
        # Use custom preprocessing during loading
        preprocessing_fn = custom_preprocessing
    
    train_dir = os.path.join(base_dir, 'training')
    test_dir = os.path.join(base_dir, 'testing')
    val_dir = os.path.join(base_dir, "validation")
    
    # Define data generators with custom preprocessing
    train_datagen = SimpleImageGenerator(
        preprocessing_function=preprocessing_fn,
        shuffle=True
    )
    val_datagen = SimpleImageGenerator(
        preprocessing_function=preprocessing_fn,
        shuffle=True
    )
    
    test_datagen = SimpleImageGenerator(
        preprocessing_function=preprocessing_fn,
        shuffle=False
    )
    
    # Load training data
    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_dataset = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load testing data
    test_dataset = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_dataset.class_indices.keys())
    
    print(f"\nDataset Summary:")
    print(f"  Total classes: {len(class_names)}")
    print(f"  Training images: {train_dataset.samples}")
    print(f"  Validation images: {val_dataset.samples}")
    print(f"  Testing images: {test_dataset.samples}")
    print("\nClass mapping:")
    for class_name, idx in train_dataset.class_indices.items():
        print(f"  {idx}: {class_name}")
    
    return train_dataset, val_dataset, test_dataset, train_dataset.class_indices

if __name__ == "__main__":
    # Check if PROCESSED_DATASET_PATH is set in .env
    if not PROCESSED_DATASET_PATH:
        print("Warning: PROCESSED_DATASET_PATH not set in .env file. Using default 'processed_dataset'.")
        processed_dir = "processed_dataset"
    else:
        processed_dir = PROCESSED_DATASET_PATH
        
    # First preprocess and save all images, then load the dataset
    train_dataset, val_dataset, test_dataset, class_indices = load_dataset_with_preprocessing(
        base_dir=processed_dir,
        img_size=(224, 224),
        batch_size=32,
        preprocess_first=True  # This enables preprocessing and saving
    )
    
    # Visualize a sample of preprocessed images (optional)
    print("Displaying sample preprocessed images...")
    
    # Get a batch from training data
    x_batch, y_batch = next(train_dataset)
    
    # Display 5 sample images
    plt.figure(figsize=(15, 10))
    for i in range(min(5, len(x_batch))):
        plt.subplot(1, 5, i+1)
        # Convert from [0,1] range back to [0,255] for display
        plt.imshow(x_batch[i])
        class_idx = np.argmax(y_batch[i])
        class_name = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]
        plt.title(f"Class: {class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save to model directory from environment
    save_path = os.path.join(MODEL_DIR, "sample_preprocessed_images.png")
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Sample images saved to {save_path}")
    
    plt.show()