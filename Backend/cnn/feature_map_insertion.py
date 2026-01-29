import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cnn.pre_trained_feature_map import (
    extract_features_from_generator,
    flatten_features,
    temporal_pooling,
    multi_view_fusion
)

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


def extract_and_fuse_features(generator):
    """Extract features from generator, flatten, pool temporally, and fuse multi-view."""
    features = extract_features_from_generator(generator, 'inference')
    flattened = flatten_features(features)
    pooled = temporal_pooling(flattened)
    fused = multi_view_fusion(pooled)
    return fused


def get_feature_vector(fused_features, object_id):

    for key in fused_features.keys():
        if object_id in key:
            return fused_features[key]['features']  # Return just the features array
    
    return None