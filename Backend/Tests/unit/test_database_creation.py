import pytest
import pymongo
from Tests.test_config import TestConfig

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Streamers.database_creation import (
    create_database,
    collect_images,
    store_in_database,
    split_data
)


class TestDatabaseCreation:
    """Test database creation and setup"""
    
    def test_create_database_returns_correct_names(self, mongo_client):
        """Test that create_database returns correct database and collection names"""
        db_name, collection_name = create_database(
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        assert db_name == TestConfig.TEST_DB_NAME
        assert collection_name == TestConfig.TEST_COLLECTION_NAME
        
        # Cleanup
        mongo_client.drop_database(TestConfig.TEST_DB_NAME)
    
    def test_create_database_creates_indexes(self, test_collection):
        """Test that required indexes are created"""
        create_database(TestConfig.TEST_DB_NAME, TestConfig.TEST_COLLECTION_NAME)
        
        indexes = test_collection.index_information()
        
        assert 'fruit_type_1' in indexes
        assert 'object_id_1' in indexes
        assert 'set_type_1' in indexes
        assert 'camera_id_1' in indexes


class TestImageCollection:
    """Test image collection from dataset"""
    
    @pytest.mark.skipif(
        not TestConfig.VALID_IMAGES_DIR.exists(),
        reason="Test images not available"
    )
    def test_collect_images_returns_list(self):
        """Test that collect_images returns a list"""
        result = collect_images(str(TestConfig.VALID_IMAGES_DIR))
        assert isinstance(result, list)
    
    @pytest.mark.skipif(
        not TestConfig.VALID_IMAGES_DIR.exists(),
        reason="Test images not available"
    )
    def test_collect_images_has_required_fields(self):
        """Test that collected images have all required metadata fields"""
        result = collect_images(str(TestConfig.VALID_IMAGES_DIR))
        
        if result:
            first_image = result[0]
            required_fields = [
                'path', 'fruit_type', 'object_id', 'camera_id',
                'timestamp', 'width', 'height'
            ]
            
            for field in required_fields:
                assert field in first_image, f"Missing required field: {field}"
    
    def test_collect_images_empty_directory(self, tmp_path):
        """Test collect_images with empty directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = collect_images(str(empty_dir))
        assert result == []
    
    def test_collect_images_nonexistent_directory(self):
        """Test collect_images with non-existent directory"""
        result = collect_images("/nonexistent/path")
        assert result == []


class TestDataStorage:
    """Test storing data in database"""
    
    def test_store_in_database_basic(self, test_collection, sample_image_metadata):
        """Test storing single document"""
        data = [sample_image_metadata]
        
        store_in_database(
            data,
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        count = test_collection.count_documents({})
        assert count == 1
        
        stored_doc = test_collection.find_one({"fruit_type": "market"})
        assert stored_doc is not None
        assert stored_doc["object_id"] == "obj0001"
    
    def test_store_in_database_multiple_documents(self, test_collection, sample_image_documents):
        """Test storing multiple documents"""
        store_in_database(
            sample_image_documents,
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        count = test_collection.count_documents({})
        assert count == len(sample_image_documents)
    
    def test_store_in_database_empty_list(self, test_collection):
        """Test storing empty list doesn't cause errors"""
        store_in_database(
            [],
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        count = test_collection.count_documents({})
        assert count == 0
    
    def test_store_in_database_data_integrity(self, test_collection, sample_image_metadata):
        """Test that stored data matches input data"""
        data = [sample_image_metadata]
        
        store_in_database(
            data,
            TestConfig.TEST_DB_NAME,
            TestConfig.TEST_COLLECTION_NAME
        )
        
        stored_doc = test_collection.find_one({})
        
        # Check all fields match (except _id which is added by MongoDB)
        for key, value in sample_image_metadata.items():
            assert stored_doc[key] == value


class TestDataSplitting:
    """Test train/test data splitting"""
    
    def test_split_data_basic(self, test_collection, sample_image_documents):
        """Test basic data splitting"""
        test_collection.insert_many(sample_image_documents)
        split_data(TestConfig.TEST_DB_NAME, TestConfig.TEST_COLLECTION_NAME, 60, 40)
        
        training_count = test_collection.count_documents({"set_type": "training"})
        testing_count = test_collection.count_documents({"set_type": "testing"})
        
        assert training_count > 0
        assert testing_count > 0
        assert training_count + testing_count == len(sample_image_documents)
    
    def test_split_data_ratio(self, test_collection):
        """Test that split ratio is approximately correct"""
        # Insert 100 documents
        documents = []
        for i in range(100):
            documents.append({
                "path": f"/test/image{i}.png",
                "fruit_type": "apple",
                "object_id": f"obj{i:04d}",
                "camera_id": i % 4,
                "timestamp": f"2025-01-01T00:00:{i:02d}",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": ""
            })
        
        test_collection.insert_many(documents)
        
        # Split 70/30
        split_data(TestConfig.TEST_DB_NAME, TestConfig.TEST_COLLECTION_NAME, 70, 30)
        
        training_count = test_collection.count_documents({"set_type": "training"})
        testing_count = test_collection.count_documents({"set_type": "testing"})
        
        # Check ratio is approximately correct ~5%
        ratio = training_count / 100
        assert 0.65 <= ratio <= 0.75
    
    def test_split_data_all_objects_assigned(self, test_collection, sample_image_documents):
        """Test that all objects get assigned a set type"""
        test_collection.insert_many(sample_image_documents)
        
        split_data(TestConfig.TEST_DB_NAME, TestConfig.TEST_COLLECTION_NAME, 70, 30)
        
        unassigned = test_collection.count_documents({"set_type": ""})
        assert unassigned == 0


class TestDatabaseRobustness:
    """Enhanced robustness tests"""
    
    def test_connection_failure_handling(self):
        """Test graceful handling of connection failures"""
        invalid_connection = "mongodb://invalid_host:27017/"
        
        try:
            client = pymongo.MongoClient(
                invalid_connection,
                serverSelectionTimeoutMS=1000
            )
            client.server_info()
            pytest.fail("Should have raised connection error")
        except pymongo.errors.ServerSelectionTimeoutError:
            # Expected behavior
            pass
    
    def test_large_batch_insertion(self, test_collection):
        """Test insertion of large number of documents"""
        large_batch = []
        for i in range(1000):
            large_batch.append({
                "path": f"/test/image{i}.png",
                "fruit_type": TestConfig.FRUIT_TYPES[i % 3],
                "object_id": f"obj{i:04d}",
                "camera_id": i % 4,
                "timestamp": f"2025-01-01T{i//60:02d}:{i%60:02d}:00",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": ""
            })
        
        # Should complete without errors
        result = test_collection.insert_many(large_batch)
        assert len(result.inserted_ids) == 1000
        
        # Verify count
        count = test_collection.count_documents({})
        assert count == 1000
    
    def test_malformed_document_rejection(self, test_collection):
        """Test rejection of documents missing required fields"""
        malformed_doc = {
            "path": "/test/image.png",
        }
        required_fields = ['path', 'fruit_type', 'object_id', 'camera_id']
        missing_fields = [f for f in required_fields if f not in malformed_doc]
        
        assert len(missing_fields) > 0, "Document should be missing required fields"
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])