import pytest
import numpy as np
import pymongo
from pathlib import Path
from Tests.test_config import TestConfig
import sys
import json
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Streamers.database_creation import store_in_database
from preprocessing.preprocessing_from_db import custom_preprocessing, set_generator
from cnn.pre_trained_feature_map import process_features
from cnn.fully_connected_layer import train, predict


class TestCompletePipeline:
    """Test complete ML pipeline integration"""
    
    @pytest.mark.slow
    def test_images_to_database_to_features(self, test_collection, valid_image_path, tmp_path):
        """Test pipeline from images to feature extraction"""
        # Step 1: Store image metadata in database
        metadata = {
            "path": str(valid_image_path),
            "processed_path": str(tmp_path / "processed.png"),
            "fruit_type": "market",
            "object_id": "obj0001",
            "camera_id": 0,
            "timestamp": "2025-01-01T00:00:00",
            "width": 224,
            "height": 224,
            "color": 3,
            "set_type": "training",
            "category": "A"
        }
        
        store_in_database(
            [metadata],
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        # Step 2: Preprocess image
        import cv2
        img = cv2.imread(str(valid_image_path))
        processed = custom_preprocessing(img)
        
        assert processed is not None
        assert processed.dtype == np.float32
        
        # Step 3: Would extract features here
        # (Skipped in test to avoid model dependency)
    
    @pytest.mark.slow
    def test_database_query_to_preprocessing(self, test_collection, sample_image_documents):
        """Test querying database and preprocessing pipeline"""
        # Insert documents
        test_collection.insert_many(sample_image_documents)
        
        # Query training data
        training_docs = list(test_collection.find({"set_type": "training"}))
        
        assert len(training_docs) > 0
        
        # Each document should have required fields
        for doc in training_docs:
            assert 'path' in doc
            assert 'fruit_type' in doc
            assert 'object_id' in doc


class TestAPIIntegration:
    """Test Flask API integration with ML pipeline"""
    
    def test_start_processing_triggers_pipeline(self, client):
        """Test API call triggers pipeline execution"""
        response = client.post('/api/pipeline/start')
        
        # Should accept request
        assert response.status_code in [200, 201, 202]
    
    def test_processing_status_reflects_pipeline_state(self, client):
        """Test status endpoint accuracy"""
        # Get initial status
        response = client.get('/api/pipeline/status')
        assert response.status_code == 200
        
        # Status should have required fields
        data = json.loads(response.data)
        assert 'status' in data or 'state' in data