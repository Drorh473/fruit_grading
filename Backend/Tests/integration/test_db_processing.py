import pytest
import numpy as np
import cv2
from pathlib import Path
from Backend.Tests.test_config import TestConfig
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Streamers.database_creation import store_in_database, split_data
from preprocessing.preprocessing_from_db import custom_preprocessing


class TestDatabaseToPreprocessing:
    """Test database query to preprocessing pipeline"""
    
    def test_fetch_and_preprocess_images(self, test_collection, valid_image_path, tmp_path):
        """Test fetching from DB and preprocessing"""
        # Step 1: Store metadata in database
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
        
        # Step 2: Query from database
        doc = test_collection.find_one({"object_id": "obj0001"})
        assert doc is not None
        
        # Step 3: Load and preprocess image
        img = cv2.imread(doc['path'])
        processed = custom_preprocessing(img)
        
        assert processed is not None
        assert processed.dtype == np.float32
    
    def test_parallel_preprocessing_multiple_cameras(self, test_collection, valid_image_path, tmp_path):
        """Test concurrent preprocessing of camera feeds"""
        # Insert multiple camera images
        metadata_list = []
        for camera_id in range(TestConfig.NUM_OF_CAMERAS):
            metadata_list.append({
                "path": str(valid_image_path),
                "processed_path": str(tmp_path / f"processed_cam{camera_id}.png"),
                "fruit_type": "market",
                "object_id": "obj0001",
                "camera_id": camera_id,
                "timestamp": "2025-01-01T00:00:00",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "training",
                "category": "A"
            })
        
        store_in_database(
            metadata_list,
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        # Query all cameras
        docs = list(test_collection.find({"object_id": "obj0001"}))
        assert len(docs) == TestConfig.NUM_OF_CAMERAS
        
        # Process all cameras
        processed_images = []
        for doc in docs:
            img = cv2.imread(doc['path'])
            processed = custom_preprocessing(img)
            processed_images.append(processed)
        
        assert len(processed_images) == TestConfig.NUM_OF_CAMERAS
 

class TestPreprocessingUpdates:
    """Test updating database with preprocessing results"""
    
    def test_update_processed_path(self, test_collection, valid_image_path, tmp_path):
        """Test updating document with processed image path"""
        # Insert original metadata
        metadata = {
            "path": str(valid_image_path),
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
        
        test_collection.insert_one(metadata)
        
        # Preprocess and save
        img = cv2.imread(str(valid_image_path))
        processed = custom_preprocessing(img)
        
        processed_path = tmp_path / "processed.png"
        processed_uint8 = (processed * 255).astype(np.uint8)
        cv2.imwrite(str(processed_path), processed_uint8)
        
        # Update database
        test_collection.update_one(
            {"object_id": "obj0001"},
            {"$set": {"processed_path": str(processed_path)}}
        )
        
        # Verify update
        doc = test_collection.find_one({"object_id": "obj0001"})
        assert "processed_path" in doc
        assert Path(doc["processed_path"]).exists()


class TestDatasetGeneration:
    """Test generating datasets from database"""
    
    def test_generate_training_set(self, test_collection, sample_image_documents):
        """Test generating training dataset"""
        # Insert documents
        test_collection.insert_many(sample_image_documents)
        
        # Assign set_type to field
        split_data(
            db_name=TestConfig.TEST_DB_NAME,
            collection_name=TestConfig.TEST_COLLECTION_NAME, 
            training_percentage=70,
            testing_percentage=30
        )

        # Query training data
        training_docs = list(test_collection.find({"set_type": "training"}))
        
        assert len(training_docs) > 0
        
        # Create paths list
        training_paths = [doc.get('processed_path') or doc.get('path') 
                         for doc in training_docs]
        
        assert len(training_paths) == len(training_docs)
    
    def test_generate_testing_set(self, test_collection, sample_image_documents):
        """Test generating testing dataset"""
        # Insert documents
        test_collection.insert_many(sample_image_documents)
        
        # Assign set_type to field
        split_data(
            db_name=TestConfig.TEST_DB_NAME,
            collection_name=TestConfig.TEST_COLLECTION_NAME, 
            training_percentage=70,
            testing_percentage=30
        )

        # Query testing data
        testing_docs = list(test_collection.find({"set_type": "testing"}))
        
        assert len(testing_docs) > 0
        
        # Create paths list
        testing_paths = [doc.get('processed_path') or doc.get('path') 
                        for doc in testing_docs]
        
        assert len(testing_paths) == len(testing_docs)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])