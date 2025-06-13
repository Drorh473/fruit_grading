import numpy as np
from collections import defaultdict

def cosine_distance(feature1, feature2):
    """Calculate cosine distance between two feature vectors"""
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    return 1 - cosine_sim

def euclidean_distance(feature1, feature2):
    """Calculate euclidean distance between two feature vectors"""
    return np.linalg.norm(feature1 - feature2)

def knn_classify(feature_vector, reference_features, reference_labels, k=3, distance_metric='cosine'):
    """
    Classify using k-nearest neighbors approach

    Returns:
        Classification result
    """
    distances = []
    
    # Calculate distance to all reference samples
    for ref_id, ref_features in reference_features.items():
        if ref_id in reference_labels:
            if distance_metric == 'cosine':
                dist = cosine_distance(feature_vector, ref_features)
            else:
                dist = euclidean_distance(feature_vector, ref_features)
            
            distances.append((dist, reference_labels[ref_id], ref_id))
    
    if not distances:
        return {
            'predicted_class': 1,
            'class_name': 'Standard',
            'confidence': 0.33,
            'nearest_neighbors': []
        }
    
    # Sort by distance and take k nearest
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Vote based on k nearest neighbors
    class_votes = defaultdict(int)
    for dist, class_label, ref_id in k_nearest:
        class_votes[class_label] += 1
    
    # Get majority vote
    predicted_class = max(class_votes, key=class_votes.get)
    confidence = class_votes[predicted_class] / k
    
    class_names = ['Premium', 'Standard', 'Market']
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'nearest_neighbors': [(dist, class_label, ref_id) for dist, class_label, ref_id in k_nearest]
    }

def classify_fused_features(fused_features, reference_features, reference_labels, k=3, distance_metric='cosine'):
    """
    Classify multiple features using k-NN approach
    
    Returns:
        Dict with classification results
    """
    results = {}
    
    for obj_id, feature_vector in fused_features.items():
        if obj_id not in reference_labels:  # Don't classify reference samples
            result = knn_classify(
                feature_vector, reference_features, reference_labels, k, distance_metric
            )
            results[obj_id] = result
    
    return results

def classify_single_feature(feature_vector, reference_features, reference_labels, k=3, distance_metric='cosine'):
    """
    Classify a single feature vector using k-NN
    
    Returns:
        Classification result
    """
    return knn_classify(feature_vector, reference_features, reference_labels, k, distance_metric)