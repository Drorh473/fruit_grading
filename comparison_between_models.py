import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)
import timm # import for the EfficientNet-Lite
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import pymongo
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from preprocessing.preprocessing_from_db import custom_preprocessing

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS', 4))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
IMAGE_SIZE = (224, 224)
TARGET_FRUITS = ["tomatoes", "oranges", "mandarins"]

def load_models():
    """Load all pre-trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models = {}
    feature_extractors = {}
    
    print("Loading pre-trained models...")
    
    # ShuffleNetV2
    shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    shufflenet = shufflenet.to(device)
    shufflenet.eval()
    models['ShuffleNetV2'] = shufflenet
    feature_extractors['ShuffleNetV2'] = nn.Sequential(*list(shufflenet.children())[:-1])
    
    # MobileNetV3
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    models['MobileNetV3'] = mobilenet
    feature_extractors['MobileNetV3'] = mobilenet.features
    
    # EfficientNet-Lite
    efficientnet_lite = timm.create_model('efficientnet_lite0', pretrained=True)
    efficientnet_lite = efficientnet_lite.to(device)
    efficientnet_lite.eval()
    models['EfficientNet-Lite'] = efficientnet_lite
    # Create feature extractor by removing classifier
    feature_extractors['EfficientNet-Lite'] = nn.Sequential(*list(efficientnet_lite.children())[:-1])
    
    print("All models loaded successfully!")
    return models, feature_extractors, device

def get_model_info(model):
    """Get model parameters and size information"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return total_params, size_mb

def preprocess_image_for_model(image_path, device, use_custom_preprocessing=True):
    """Preprocess image using your custom preprocessing pipeline"""
    try:
        if use_custom_preprocessing:
            # Use your custom preprocessing from preprocessing_from_db.py
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply your custom preprocessing
            processed_img = custom_preprocessing(img)
            
            # Convert back to PIL Image for torchvision transforms
            if processed_img.max() <= 1.0:
                processed_img = (processed_img * 255).astype(np.uint8)
            img_pil = Image.fromarray(processed_img)
        else:
            img_pil = Image.open(image_path).convert('RGB')
        
        # Standard transforms for all models
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_pil)
        return img_tensor.unsqueeze(0).to(device)
    
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def load_test_data_from_db():
    """Load test data from your MongoDB database structure"""
    client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db["images"]
    
    test_data = []
    
    print("Loading test data from database...")
    
    # Get test images for target fruits with orientation information
    for fruit_type in TARGET_FRUITS:
        documents = list(collection.find({
            "set_type": "testing",
            "fruit_type": fruit_type
        }))
        
        print(f"Found {len(documents)} test images for {fruit_type}")
        
        for doc in documents:
            # Use processed_path if available, otherwise use original path
            img_path = doc.get("processed_path", doc.get("path"))
            
            if img_path and os.path.exists(img_path):
                test_data.append({
                    'path': img_path,
                    'fruit_type': fruit_type,
                    'object_id': doc.get('object_id'),
                    'camera_id': doc.get('camera_id', 1),
                    'timestamp': doc.get('timestamp')
                })
    
    client.close()
    print(f"Total test images loaded: {len(test_data)}")
    return test_data

def extract_features_for_model(model_name, feature_extractor, test_data, device, batch_size=BATCH_SIZE):
    """Extract features using the specified model"""
    print(f"Extracting features using {model_name}...")
    
    all_features = []
    processing_times = []
    valid_indices = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Feature extraction ({model_name})"):
        batch_data = test_data[i:i+batch_size]
        batch_tensors = []
        batch_indices = []
        
        # Load batch
        for j, data in enumerate(batch_data):
            tensor = preprocess_image_for_model(data['path'], device)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_indices.append(i + j)
        
        if not batch_tensors:
            continue
        
        # Stack tensors
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # Extract features
        start_time = time.time()
        with torch.no_grad():
            features = feature_extractor(batch_tensor)
            
            # Handle different output shapes from different models
            if len(features.shape) > 2:
                # Apply global average pooling if needed
                features = torch.mean(features, dim=[2, 3]) if len(features.shape) == 4 else features
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time / len(batch_tensors))  # Per image
        
        # Move to CPU and store
        features = features.cpu().numpy()
        all_features.extend(features)
        valid_indices.extend(batch_indices)
    
    return np.array(all_features), processing_times, valid_indices

def measure_inference_time(model, test_data, device, num_runs=100):
    """Measure inference time for a model on your actual data"""
    # Select random subset for timing
    selected_indices = np.random.choice(len(test_data), min(num_runs, len(test_data)), replace=False)
    selected_data = [test_data[i] for i in selected_indices]
    
    inference_times = []
    
    print(f"Measuring inference time on {len(selected_data)} images...")
    for data in tqdm(selected_data, desc="Timing inference"):
        img_tensor = preprocess_image_for_model(data['path'], device)
        if img_tensor is None:
            continue
        
        # Warm up GPU
        with torch.no_grad():
            _ = model(img_tensor)
        
        # Measure time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(img_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.array(inference_times)

def run_model_comparison():
    """Run complete comparison between models using your data structure"""
    print("Starting model comparison using your fruit dataset structure...")
    
    # Load models
    models, feature_extractors, device = load_models()
    
    # Load your test data
    test_data = load_test_data_from_db()
    
    if not test_data:
        print("No test data found! Check your database connection and data.")
        return None
    
    # Results storage
    results = {
        'model': [],
        'inference_time_avg_ms': [],
        'inference_time_std_ms': [],
        'feature_extraction_time_ms': [],
        'total_params': [],
        'model_size_mb': [],
        'num_test_images': []
    }
    
    model_names = ['ShuffleNetV2', 'MobileNetV3', 'EfficientNet-Lite']
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        model = models[model_name]
        feature_extractor = feature_extractors[model_name]
        
        # Get model info
        total_params, model_size_mb = get_model_info(model)
        
        # Measure inference time
        inference_times = measure_inference_time(model, test_data, device)
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        # Extract features and measure feature extraction time
        features, feature_times, valid_indices = extract_features_for_model(
            model_name, feature_extractor, test_data, device
        )
        avg_feature_time = np.mean(feature_times) * 1000 if feature_times else 0  # Convert to ms
        
        # Store results
        results['model'].append(model_name)
        results['inference_time_avg_ms'].append(avg_inference_time)
        results['inference_time_std_ms'].append(std_inference_time)
        results['feature_extraction_time_ms'].append(avg_feature_time)
        results['total_params'].append(total_params)
        results['model_size_mb'].append(model_size_mb)
        results['num_test_images'].append(len(valid_indices))
        
        # Print intermediate results
        print(f"Results for {model_name}:")
        print(f"  Inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"  Feature extraction time: {avg_feature_time:.2f} ms")
        print(f"  Parameters: {total_params:,}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Test images processed: {len(valid_indices)}")
    
    return results

def save_and_visualize_results(results):
    """Save results and create visualizations"""
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = os.path.join(MODEL_DIR, 'fruit_model_comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Fruit Grading Model Comparison: ShuffleNetV2 vs MobileNetV3 vs EfficientNet-Lite', fontsize=20)
    
    # 1. Inference Time
    axes[0, 0].bar(results_df['model'], results_df['inference_time_avg_ms'], 
                   yerr=results_df['inference_time_std_ms'], capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 0].set_title('Average Inference Time')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Feature Extraction Time
    axes[0, 1].bar(results_df['model'], results_df['feature_extraction_time_ms'], 
                   color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 1].set_title('Feature Extraction Time')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Model Parameters
    axes[1, 1].bar(results_df['model'], results_df['total_params'] / 1e6, 
                   color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 1].set_title('Model Parameters')
    axes[1, 1].set_ylabel('Parameters (Millions)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 4. Model Size
    axes[1, 0].bar(results_df['model'], results_df['model_size_mb'], 
                   color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 0].set_title('Model Size')
    axes[1, 0].set_ylabel('Size (MB)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(MODEL_DIR, 'fruit_model_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_path}")
    plt.show()
    
    return results_df

def print_detailed_summary(results_df):
    """Print a detailed summary of the comparison"""
    # Create a nicely formatted table
    display_df = results_df.copy()
    display_df = display_df.round(3)
    display_df['total_params'] = display_df['total_params'].apply(lambda x: f"{x/1e6:.1f}M")
    
    print(display_df.to_string(index=False))

def main():
    """Main function to run the complete model comparison"""
    print("="*80)
    print("FRUIT GRADING PROJECT - MODEL COMPARISON")
    print("="*80)
    print("Comparing: ShuffleNetV2, MobileNetV3, EfficientNet-Lite")
    print(f"Target fruits: {TARGET_FRUITS}")
    print(f"Using your custom preprocessing pipeline")
    
    # Check if required paths exist
    if not os.path.exists(PROCESSED_DATASET_PATH or ""):
        print(f"\nWarning: PROCESSED_DATASET_PATH not found: {PROCESSED_DATASET_PATH}")
        print("Make sure your dataset is processed and paths are correct in .env file")
        return None
    
    # Run comparison
    results = run_model_comparison()
    
    if results is None:
        print("Comparison failed. Check your data and configuration.")
        return None
    
    # Save and visualize results
    results_df = save_and_visualize_results(results)
    
    # Print detailed summary
    print_detailed_summary(results_df)
    
    return results_df

if __name__ == "__main__":
    results = main()