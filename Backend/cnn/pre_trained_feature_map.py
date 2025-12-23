import numpy as np
import torch
import os
from tqdm import tqdm
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Define label mapping as a constant
LABEL_DICT = {
    'market': 0,
    'standard': 1,
    'unknown': 2
}

def load_model():
    """
    Load ShuffleNetV2 model from torchvision
    
    Returns:
        model: The complete ShuffleNetV2 model
        feature_extractor: Model without the classifier layer
        device: Device the model is loaded on
    """
    # Set device (use CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
  
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create feature extractor by removing classifier
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    return model, feature_extractor, device

def extract_features_from_generator(generator, set_type):
    """
    Extract features from a batch generator
    Args:
        generator: Generator function that yields batches of (images, metadata)
        set_type: 'training' or 'testing'
    Returns:
        Dictionary mapping keys to feature data with labels
    """
    # Load model once
    _, feature_extractor, device = load_model()
    
    print(f"Extracting features from {set_type} generator...")
    
    feature_map = {}
    batch_count = 0
    image_count = 0
    
    # Check if generator has num_batches attribute for progress bar
    num_batches = getattr(generator, 'num_batches', None)
    
    # Process each batch from generator
    for batch in tqdm(generator(), total=num_batches, desc=f"Processing {set_type}"):
        batch_count += 1
        
        # Handle different batch formats
        if isinstance(batch, tuple):
            images, metadata = batch
        else:
            images = batch
            metadata = None
        
        # Check if we actually have images
        if images is None or len(images) == 0:
            print(f"Warning: Empty batch {batch_count}")
            continue
        
        # Check if we have metadata
        if metadata is None or len(metadata) == 0:
            print(f"Warning: No metadata for batch {batch_count}")
            continue
        
        tqdm.write(f"Processing batch {batch_count} with {len(images)} images")
        
        # Process each image in batch
        for idx, image in enumerate(images):
            try:
                # Convert image to tensor if needed
                if not isinstance(image, torch.Tensor):
                    # Normalize if needed (image should be 0-255 or 0-1)
                    if image.max() > 1.0:
                        image = image.astype(np.float32) / 255.0
                    
                    # Convert to tensor (H, W, C) -> (C, H, W)
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                    image_tensor = image_tensor.to(device)
                else:
                    image_tensor = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
                
                # Extract features
                with torch.no_grad():
                    features = feature_extractor(image_tensor)
                
                # Convert to numpy and reshape (1, C, H, W) -> (H, W, C)
                features = features.cpu().numpy()
                features = np.transpose(features, (0, 2, 3, 1))[0]
                
                # Create key
                meta = metadata[idx]
                fruit_type = meta.get('fruit_type', 'unknown')
                key = f"{fruit_type}_{meta.get('object_id')}_{meta.get('camera_id')}"
                    
                if key not in feature_map:
                    feature_map[key] = []
                    
                feature_map[key].append({
                    'features': features,
                    'timestamp': meta.get('timestamp'),
                    'label': LABEL_DICT.get(fruit_type, 2),  # Store label here
                    'fruit_type': fruit_type
                })
    
                image_count += 1
                
            except Exception as e:
                print(f"Error processing image {idx} in batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Extracted features from {image_count} images in {batch_count} batches")
    
    return feature_map

def flatten_features(feature_map):
    """
    Flatten spatial dimensions
    Args:
        feature_map: Dictionary with lists of features (with labels)
    Returns:
        Dictionary with flattened features (preserving labels)
    """
    flattened = {}
    
    for key, feature_list in feature_map.items():
        for idx, item in enumerate(feature_list):
            h, w, c = item['features'].shape
            flat_features = item['features'].reshape(h * w * c)
            
            timestep_key = f"{key}_t{idx}"
            flattened[timestep_key] = {
                'features': flat_features,
                'group_key': key,
                'label': item.get('label', 0),
                'fruit_type': item.get('fruit_type', 'unknown')
            }
    
    return flattened

def temporal_pooling(flattened_features):
    """
    Average features across time
    Args:
        flattened_features: Dictionary with flattened features per timestep
    Returns:
        Dictionary with pooled features (preserving labels)
    """
    # Group by key
    grouped = {}
    for timestep_key, data in flattened_features.items():
        group_key = data['group_key']
        if group_key not in grouped:
            grouped[group_key] = {
                'features': [],
                'label': data.get('label', 0),
                'fruit_type': data.get('fruit_type', 'unknown') 
            }
        grouped[group_key]['features'].append(data['features'])
    
    # Average
    pooled = {}
    for key, data in grouped.items():
        features_list = data['features']
        if len(features_list) == 1:
            pooled_features = features_list[0]
        else:
            pooled_features = np.mean(np.stack(features_list), axis=0)
        
        pooled[key] = {
            'features': pooled_features,
            'label': data['label'],
            'fruit_type': data['fruit_type']
        }
    
    print(f"Temporal pooling: {len(pooled)} groups")
    return pooled

def multi_view_fusion(pooled_vectors, target_views=4):
    """
    Concatenate features from different cameras
    Ensures consistent output dimension by padding or truncating views
    
    Args:
        pooled_vectors: Dictionary with pooled features
        target_views: Target number of views (default 4 for 4 cameras)
    Returns:
        Dictionary with fused features (preserving labels)
    """
    # Group by object (remove camera suffix)
    grouped = {}
    for key, data in pooled_vectors.items():
        # Try to extract base object ID (remove camera_id)
        parts = key.rsplit('_', 1)  # Split from right
        base_key = parts[0] if len(parts) > 1 else key
        
        if base_key not in grouped:
            grouped[base_key] = {
                'features': [],
                'label': data.get('label', 0),
                'fruit_type': data.get('fruit_type', 'unknown') 
            }
        grouped[base_key]['features'].append(data['features'])
    
    # Concatenate with padding/truncating
    fused = {}
    feature_dim_per_view = None
    
    for key, data in grouped.items():
        vectors = data['features']
        
        # Get feature dimension from first vector
        if feature_dim_per_view is None:
            feature_dim_per_view = vectors[0].shape[0]
        
        # Pad or truncate to target number of views
        if len(vectors) < target_views:
            # Pad with zeros
            padding_needed = target_views - len(vectors)
            zero_vector = np.zeros(feature_dim_per_view, dtype=vectors[0].dtype)
            vectors.extend([zero_vector] * padding_needed)
        elif len(vectors) > target_views:
            # Truncate (keep first N views)
            vectors = vectors[:target_views]
        
        # Concatenate
        fused_features = np.concatenate(vectors)
        
        fused[key] = {
            'features': fused_features,
            'label': data['label'],
            'fruit_type': data['fruit_type']
        }
    
    print(f"Multi-view fusion: {len(fused)} objects")
    if fused:
        sample_dim = next(iter(fused.values()))['features'].shape[0]
        print(f"Feature dimension: {sample_dim:,} features ({target_views} views × {feature_dim_per_view:,})")
    
    return fused
def process_features(generator, set_type):
    """
    Extract → Flatten → Pool → Fuse 
    Args:
        generator: Generator for images
        set_type: 'training' or 'testing'
    Returns:
        Dictionary with final fused feature vectors (with labels)
    """ 
    # Extract features
    print("\n✓ Extracting features...")
    features = extract_features_from_generator(generator, set_type)
    
    # Flatten
    print("\n✓ Flattening features...")
    flattened = flatten_features(features)
    
    # Temporal pooling
    print("\n✓ Temporal pooling...")
    pooled = temporal_pooling(flattened)
    
    # Multi-view fusion
    print("\n✓ Multi-view fusion...")
    fused = multi_view_fusion(pooled)
    
    return fused