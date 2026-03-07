"""ShuffleNetV2 feature extraction with multi-view fusion."""
import numpy as np
import torch
from tqdm import tqdm
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

LABEL_DICT = {
    'market': 0,
    'standard': 1,
    'unknown': 2
}

_MODEL_CACHE = {
    'model': None,
    'feature_extractor': None,
    'device': None
}


def load_model():
    """Load and cache ShuffleNetV2 model."""
    if _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['model'], _MODEL_CACHE['feature_extractor'], _MODEL_CACHE['device']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['feature_extractor'] = feature_extractor
    _MODEL_CACHE['device'] = device

    return model, feature_extractor, device

def extract_features_from_generator(generator, set_type):
    """Extract features from batch generator using batched inference."""
    _, feature_extractor, device = load_model()

    feature_map = {}
    batch_count = 0
    image_count = 0
    num_batches = getattr(generator, 'num_batches', None)

    for batch in tqdm(generator(), total=num_batches, desc=f"Processing {set_type}", disable=(set_type == 'inference')):
        batch_count += 1

        if isinstance(batch, tuple):
            images, metadata = batch
        else:
            images = batch
            metadata = None

        if images is None or len(images) == 0:
            continue

        if metadata is None or len(metadata) == 0:
            continue

        try:
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

            batch_tensor = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                batch_features = feature_extractor(batch_tensor)

            batch_features = batch_features.cpu().numpy()
            batch_features = np.transpose(batch_features, (0, 2, 3, 1))

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

        except Exception:
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
    """Flatten spatial dimensions of feature maps."""
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
    """Average features across time steps."""
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
    """Average-pool features from multiple camera views."""
    grouped = {}
    for key, data in pooled_vectors.items():
        parts = key.rsplit('_', 1)
        base_key = parts[0] if len(parts) > 1 else key

        if base_key not in grouped:
            grouped[base_key] = {
                'features': [],
                'label': data.get('label', 0),
                'fruit_type': data.get('fruit_type', 'unknown')
            }
        grouped[base_key]['features'].append(data['features'])

    fused = {}
    feature_dim_per_view = None

    for key, data in grouped.items():
        vectors = data['features']

        if feature_dim_per_view is None:
            feature_dim_per_view = vectors[0].shape[0]

        if len(vectors) > target_views:
            vectors = vectors[:target_views]

        fused_features = np.mean(np.stack(vectors), axis=0)

        fused[key] = {
            'features': fused_features,
            'label': data['label'],
            'fruit_type': data['fruit_type']
        }

    return fused


def process_features(generator, set_type):
    """Complete feature processing pipeline: extract, flatten, pool, fuse."""
    features = extract_features_from_generator(generator, set_type)
    flattened = flatten_features(features)
    pooled = temporal_pooling(flattened)
    fused = multi_view_fusion(pooled)
    return fused