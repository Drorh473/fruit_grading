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

# Model cache - loaded once, reused for all requests
_MODEL_CACHE = {
    'model': None,
    'feature_extractor': None,
    'device': None
}

def load_model():
    """
    Load ShuffleNetV2 model from torchvision (cached after first load)

    Returns:
        model: The complete ShuffleNetV2 model
        feature_extractor: Model without the classifier layer
        device: Device the model is loaded on
    """
    # Return cached model if available
    if _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['model'], _MODEL_CACHE['feature_extractor'], _MODEL_CACHE['device']

    print("Loading ShuffleNetV2 model (first time only)...")

    # Set device (use CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    # Move model to device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Create feature extractor by removing classifier
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    # Cache the model
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['feature_extractor'] = feature_extractor
    _MODEL_CACHE['device'] = device

    print(f"Model loaded on {device}")

    return model, feature_extractor, device

def extract_features_from_generator(generator, set_type):
    """
    Extract features from a batch generator using BATCHED inference for speed.
    Args:
        generator: Generator function that yields batches of (images, metadata)
        set_type: 'training' or 'testing'
    Returns:
        Dictionary mapping keys to feature data with labels
    """
    # Load model once
    _, feature_extractor, device = load_model()

    feature_map = {}
    batch_count = 0
    image_count = 0

    # Check if generator has num_batches attribute for progress bar
    num_batches = getattr(generator, 'num_batches', None)

    # Process each batch from generator
    for batch in tqdm(generator(), total=num_batches, desc=f"Processing {set_type}", disable=(set_type == 'inference')):
        batch_count += 1

        # Handle different batch formats
        if isinstance(batch, tuple):
            images, metadata = batch
        else:
            images = batch
            metadata = None

        # Check if we actually have images
        if images is None or len(images) == 0:
            continue

        # Check if we have metadata
        if metadata is None or len(metadata) == 0:
            continue

        try:
            # BATCH PROCESSING: Convert all images to tensor at once
            batch_tensors = []
            valid_indices = []

            for idx, image in enumerate(images):
                if not isinstance(image, torch.Tensor):
                    if image.max() > 1.0:
                        image = image.astype(np.float32) / 255.0
                    # Convert (H, W, C) -> (C, H, W)
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                else:
                    image_tensor = image
                batch_tensors.append(image_tensor)
                valid_indices.append(idx)

            if not batch_tensors:
                continue

            # Stack into single batch tensor and move to device
            batch_tensor = torch.stack(batch_tensors).to(device)

            # Extract features for entire batch at once
            with torch.no_grad():
                batch_features = feature_extractor(batch_tensor)

            # Convert to numpy: (N, C, H, W) -> (N, H, W, C)
            batch_features = batch_features.cpu().numpy()
            batch_features = np.transpose(batch_features, (0, 2, 3, 1))

            # Process results
            for i, idx in enumerate(valid_indices):
                features = batch_features[i]
                meta = metadata[idx]
                fruit_type = meta.get('fruit_type', 'unknown')
                key = f"{fruit_type}_{meta.get('object_id')}_{meta.get('camera_id')}"

                if key not in feature_map:
                    feature_map[key] = []

                feature_map[key].append({
                    'features': features,
                    'timestamp': meta.get('timestamp'),
                    'label': LABEL_DICT.get(fruit_type, 2),
                    'fruit_type': fruit_type
                })
                image_count += 1

        except Exception as e:
            # Fall back to individual processing if batch fails
            for idx, image in enumerate(images):
                try:
                    if not isinstance(image, torch.Tensor):
                        if image.max() > 1.0:
                            image = image.astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                        image_tensor = image_tensor.to(device)
                    else:
                        image_tensor = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)

                    with torch.no_grad():
                        features = feature_extractor(image_tensor)

                    features = features.cpu().numpy()
                    features = np.transpose(features, (0, 2, 3, 1))[0]

                    meta = metadata[idx]
                    fruit_type = meta.get('fruit_type', 'unknown')
                    key = f"{fruit_type}_{meta.get('object_id')}_{meta.get('camera_id')}"

                    if key not in feature_map:
                        feature_map[key] = []

                    feature_map[key].append({
                        'features': features,
                        'timestamp': meta.get('timestamp'),
                        'label': LABEL_DICT.get(fruit_type, 2),
                        'fruit_type': fruit_type
                    })
                    image_count += 1
                except:
                    continue

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
    
    return pooled

def multi_view_fusion(pooled_vectors, target_views=4):
    """
    Average-pool features from different cameras.
    Uses only the available views (ignores missing ones) to avoid
    diluting the signal with zero-padded vectors.

    Args:
        pooled_vectors: Dictionary with pooled features
        target_views: Target number of views (default 4 for 4 cameras)
    Returns:
        Dictionary with fused features (preserving labels)
    """
    # Group by object
    grouped = {}
    for key, data in pooled_vectors.items():
        # Try to extract base object ID
        parts = key.rsplit('_', 1)  # Split from right
        base_key = parts[0] if len(parts) > 1 else key

        if base_key not in grouped:
            grouped[base_key] = {
                'features': [],
                'label': data.get('label', 0),
                'fruit_type': data.get('fruit_type', 'unknown')
            }
        grouped[base_key]['features'].append(data['features'])

    # Average-pool across views
    fused = {}
    feature_dim_per_view = None

    for key, data in grouped.items():
        vectors = data['features']

        # Get feature dimension from first vector
        if feature_dim_per_view is None:
            feature_dim_per_view = vectors[0].shape[0]

        # Truncate if more views than expected
        if len(vectors) > target_views:
            vectors = vectors[:target_views]

        # Average-pool across all available views
        fused_features = np.mean(np.stack(vectors), axis=0)

        fused[key] = {
            'features': fused_features,
            'label': data['label'],
            'fruit_type': data['fruit_type']
        }

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
    print("\nExtracting features...")
    features = extract_features_from_generator(generator, set_type)
    
    # Flatten
    print("\nFlattening features...")
    flattened = flatten_features(features)
    
    # Temporal pooling
    print("\nTemporal pooling...")
    pooled = temporal_pooling(flattened)
    
    # Multi-view fusion
    print("\nMulti-view fusion...")
    fused = multi_view_fusion(pooled)
    
    return fused