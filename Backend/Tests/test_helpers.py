import numpy as np
import cv2
from pathlib import Path


def create_test_image(size=(224, 224, 3), color='random'):
    """Create a test image
    
    Args:
        size: Image dimensions (height, width, channels)
        color: 'random', 'red', 'green', 'blue', 'black', 'white'
    
    Returns:
        numpy array representing image
    """
    if color == 'random':
        return np.random.randint(0, 255, size, dtype=np.uint8)
    elif color == 'red':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel
        return img
    elif color == 'green':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 1] = 255  # Green channel
        return img
    elif color == 'blue':
        img = np.zeros(size, dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel
        return img
    elif color == 'black':
        return np.zeros(size, dtype=np.uint8)
    elif color == 'white':
        return np.ones(size, dtype=np.uint8) * 255


def save_test_image(path, size=(224, 224, 3), color='random'):
    """Create and save a test image
    
    Args:
        path: Path to save image
        size: Image dimensions
        color: Image color
    """
    img = create_test_image(size, color)
    cv2.imwrite(str(path), img)
    return path


def create_test_dataset(output_dir, num_images=10, num_cameras=4):
    """Create a complete test dataset
    
    Args:
        output_dir: Directory to save images
        num_images: Number of fruit objects
        num_cameras: Number of camera angles
    
    Returns:
        List of image metadata dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    fruit_types = ['market', 'standard', 'premium']
    
    for i in range(num_images):
        fruit_type = fruit_types[i % len(fruit_types)]
        object_id = f"obj{i:04d}"
        
        for camera_id in range(num_cameras):
            # Create directory structure
            camera_dir = output_dir / fruit_type / f"camera_{camera_id}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Create image
            img_name = f"{object_id}_cam{camera_id}.png"
            img_path = camera_dir / img_name
            
            save_test_image(img_path)
            
            # Create metadata
            metadata = {
                "path": str(img_path),
                "fruit_type": fruit_type,
                "object_id": object_id,
                "camera_id": camera_id,
                "timestamp": f"2025-01-01T00:{i:02d}:{camera_id:02d}",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": "A"
            }
            
            metadata_list.append(metadata)
    
    return metadata_list


def compare_images(img1, img2, tolerance=1e-5):
    """Compare two images for similarity
    
    Args:
        img1: First image
        img2: Second image
        tolerance: Maximum allowed difference
    
    Returns:
        Boolean indicating if images are similar
    """
    if img1.shape != img2.shape:
        return False
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = np.max(diff)
    
    return max_diff < tolerance


def assert_valid_image_metadata(metadata):
    """Assert that image metadata has all required fields

    Args:
        metadata: Dictionary of image metadata
    """
    required_fields = [
        'path', 'fruit_type', 'object_id', 'camera_id',
        'timestamp', 'width', 'height'
    ]

    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"


import json


def assert_json_response(response, expected_status=200):
    """Assert response is valid JSON with expected status code.

    Args:
        response: Flask test client response
        expected_status: Expected HTTP status code

    Returns:
        Parsed JSON data
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}"
    assert 'application/json' in response.content_type, \
        f"Expected JSON content type, got {response.content_type}"

    try:
        return json.loads(response.data)
    except json.JSONDecodeError:
        raise AssertionError("Response data is not valid JSON")


def assert_has_fields(data, required_fields, context="response"):
    """Assert dictionary contains all required fields.

    Args:
        data: Dictionary to check
        required_fields: List of required field names
        context: Context string for error messages
    """
    for field in required_fields:
        assert field in data, f"Missing '{field}' in {context}"


def assert_field_types(data, field_types, context="response"):
    """Assert fields have expected types.

    Args:
        data: Dictionary to check
        field_types: Dict mapping field names to expected types (or tuples of types)
        context: Context string for error messages
    """
    for field, expected_type in field_types.items():
        if field in data:
            assert isinstance(data[field], expected_type), \
                f"Field '{field}' in {context} should be {expected_type}, got {type(data[field])}"


def assert_value_in_range(value, min_val, max_val, field_name="value"):
    """Assert value is within expected range.

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name for error messages
    """
    assert min_val <= value <= max_val, \
        f"{field_name} should be between {min_val} and {max_val}, got {value}"


def check_endpoint_returns_json(client, endpoint):
    """Check that an endpoint returns valid JSON.

    Args:
        client: Flask test client
        endpoint: API endpoint path

    Returns:
        Parsed JSON data if successful
    """
    response = client.get(endpoint)
    return assert_json_response(response)


def check_endpoints_error_resilience(client, endpoints):
    """Check that multiple endpoints handle errors gracefully.

    Args:
        client: Flask test client
        endpoints: List of endpoint paths

    Returns:
        Dict mapping endpoints to their responses
    """
    results = {}
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200, \
            f"Endpoint {endpoint} should return 200, got {response.status_code}"
        results[endpoint] = json.loads(response.data) if response.data else None
    return results


def assert_pagination_response(data, max_limit=None):
    """Assert response has valid pagination fields.

    Args:
        data: Response data
        max_limit: Maximum allowed limit value
    """
    assert_has_fields(data, ['total', 'limit', 'offset'], "pagination response")
    assert isinstance(data['total'], int)
    assert isinstance(data['limit'], int)
    assert isinstance(data['offset'], int)
    assert data['total'] >= 0
    assert data['limit'] >= 0
    assert data['offset'] >= 0

    if max_limit:
        assert data['limit'] <= max_limit


def assert_list_items_have_fields(items, required_fields, context="list item"):
    """Assert all items in a list have required fields.

    Args:
        items: List of dictionaries
        required_fields: List of required field names
        context: Context string for error messages
    """
    for i, item in enumerate(items):
        for field in required_fields:
            assert field in item, f"Missing '{field}' in {context} {i}"
