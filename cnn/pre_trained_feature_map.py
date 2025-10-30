import os
import numpy as np
import torch
import time
from tqdm import tqdm
from cnn.pre_trained_cnn import load_model, extract_features
from cnn.activation_functions import relu, softmax


def extract_features_from_generator(generator, set_type):
    """
    Extract features from a batch generator
    
    Args:
        generator: Generator that yields batches of (images, metadata) or just images
        set_type: 'training' or 'testing'
        
    Returns:
        Dictionary mapping keys to feature data
    """
    # Load model once
    _, feature_extractor, device = load_model()
    
    print(f"Extracting features from {set_type} generator...")
    
    feature_map = {}
    batch_count = 0
    image_count = 0
    
    # Process each batch from generator
    for batch in tqdm(generator(), desc=f"Processing {set_type}"):
        batch_count += 1
        
        # Handle different batch formats
        if isinstance(batch, tuple):
            images, metadata = batch
        else:
            images = batch
            metadata = None
        
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
                key = f"{meta.get('fruit_type')}_{meta.get('object_id')}_{meta.get('camera_id')}"
                    
                if key not in feature_map:
                    feature_map[key] = []
                    
                feature_map[key].append({
                        'features': features,
                        'timestamp': meta.get('timestamp')
                })
    
                image_count += 1
                
            except Exception as e:
                print(f"Error processing image {idx} in batch {batch_count}: {e}")
                continue
    
    print(f"Extracted features from {image_count} images in {batch_count} batches")
    
    return feature_map

def flatten_features(feature_map):
    """
    Flatten spatial dimensions
    
    Args:
        feature_map: Dictionary with lists of features
        
    Returns:
        Dictionary with flattened features
    """
    flattened = {}
    
    for key, feature_list in feature_map.items():
        for idx, item in enumerate(feature_list):
            h, w, c = item['features'].shape
            flat_features = item['features'].reshape(h * w * c)
            
            timestep_key = f"{key}_t{idx}"
            flattened[timestep_key] = {
                'features': flat_features,
                'group_key': key
            }
    
    return flattened

def temporal_pooling(flattened_features):
    """
    Average features across time
    
    Args:
        flattened_features: Dictionary with flattened features per timestep
        
    Returns:
        Dictionary with pooled features
    """
    # Group by key
    grouped = {}
    for timestep_key, data in flattened_features.items():
        group_key = data['group_key']
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(data['features'])
    
    # Average
    pooled = {}
    for key, features_list in grouped.items():
        if len(features_list) == 1:
            pooled[key] = features_list[0]
        else:
            pooled[key] = np.mean(np.stack(features_list), axis=0)
    
    print(f"Temporal pooling: {len(pooled)} groups")
    return pooled

def multi_view_fusion(pooled_vectors):
    """
    Concatenate features from different cameras
    
    Args:
        pooled_vectors: Dictionary with pooled features
        
    Returns:
        Dictionary with fused features
    """
    # Group by object (remove camera suffix)
    grouped = {}
    for key, features in pooled_vectors.items():
        # Try to extract base object ID
        parts = key.rsplit('_', 1)  # Split from right
        base_key = parts[0] if len(parts) > 1 else key
        
        if base_key not in grouped:
            grouped[base_key] = []
        grouped[base_key].append(features)
    
    # Concatenate
    fused = {}
    for key, vectors in grouped.items():
        if len(vectors) == 1:
            fused[key] = vectors[0]
        else:
            fused[key] = np.concatenate(vectors)
    
    print(f"Multi-view fusion: {len(fused)} objects")
    if fused:
        sample_dim = next(iter(fused.values())).shape[0]
        avg_views = sum(len(v) for v in grouped.values()) / len(grouped)
        print(f"Feature dimension: {sample_dim}")
        print(f"Average views per object: {avg_views:.2f}")
    
    return fused

def process_features(train_generator, test_generator):
    """
    Complete pipeline: Extract → Flatten → Pool → Fuse → Activate
    
    Args:
        train_generator: Generator for training images
        test_generator: Generator for testing images
        
    Returns:
        Dictionary with final fused feature vectors
    """ 
    # 1. Extract features
    print("\n1. Extracting features...")
    train_features = extract_features_from_generator(train_generator, 'training')
    test_features = extract_features_from_generator(test_generator, 'testing')
    
    # Combine
    all_features = {**train_features, **test_features}
    print(f"\nTotal feature groups: {len(all_features)}")
    
    # 2. Flatten
    print("\n2. Flattening features...")
    flattened = flatten_features(all_features)
    
    # 3. Temporal pooling
    print("\n3. Temporal pooling...")
    pooled = temporal_pooling(flattened)
    
    # 4. Multi-view fusion
    print("\n4. Multi-view fusion...")
    fused = multi_view_fusion(pooled)
    
    # 5. Activation functions (optional)
    print("\n5. Applying activation functions...")
    fused = relu(fused)
    fused = softmax(fused)
    print("Activation functions applied") 
    return fused