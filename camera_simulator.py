import time
import random
import cv2
import numpy as np
import threading
import queue
from pymongo import MongoClient

class CameraSimulator:
    """
    Simulates a camera stream by retrieving images from a MongoDB database
    at a fixed frame rate (e.g., 60 FPS).
    """
    def __init__(self, db_name="fruit_grading", collection_name="images", 
                 fps=300, buffer_size=100, random_order=True):
        """
        Initialize the camera simulator.
        
        Args:
            db_name: MongoDB database name
            collection_name: MongoDB collection name
            fps: Frames per second to simulate (default: 60 FPS)
            buffer_size: Size of the buffer queue
            random_order: Whether to return images in random order
        """
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.random_order = random_order
        self.running = False
        self.buffer = queue.Queue(maxsize=buffer_size)
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB database: {db_name}.{collection_name}")
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Get all document IDs from the database
        self.doc_ids = [doc["_id"] for doc in self.collection.find({}, {"_id": 1})]
        if not self.doc_ids:
            raise ValueError("No documents found in the database!")
        
        print(f"Found {len(self.doc_ids)} images in the database")
        self.current_index = 0
        
        # Thread for background loading
        self.thread = None
        
    def start(self):
        """Start the camera simulator thread"""
        if self.thread is not None and self.thread.is_alive():
            print("Camera simulator is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker)
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera simulator started at {self.fps} FPS")
    
    def stop(self):
        """Stop the camera simulator thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("Camera simulator stopped")
    
    def _stream_worker(self):
        """Background thread that loads images into the buffer"""
        # Create a copy of the IDs for shuffling
        ids = self.doc_ids.copy()
        if self.random_order:
            random.shuffle(ids)
        
        start_time = time.time()
        frames_processed = 0
        
        while self.running:
            # Reset index if needed
            if self.current_index >= len(ids):
                self.current_index = 0
                if self.random_order:
                    random.shuffle(ids)
                
                # Print FPS statistics
                elapsed = time.time() - start_time
                if elapsed > 0:
                    actual_fps = frames_processed / elapsed
                    print(f"Processed {frames_processed} frames in {elapsed:.2f}s ({actual_fps:.2f} FPS)")
                
                start_time = time.time()
                frames_processed = 0
            
            # Get the next document ID
            doc_id = ids[self.current_index]
            self.current_index += 1
            
            try:
                # Retrieve the document from MongoDB
                doc = self.collection.find_one({"_id": doc_id})
                if doc is None:
                    continue
                
                # Check if we have a path
                if "path" not in doc or not doc["path"]:
                    continue
                
                # Load the image
                img = cv2.imread(doc["path"])
                if img is None:
                    continue
                
                # Convert to RGB format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Create metadata dictionary
                metadata = {k: v for k, v in doc.items() if k != "_id"}
                
                # Add to buffer if not full
                if not self.buffer.full():
                    self.buffer.put((img, metadata))
                    frames_processed += 1
                
                # Sleep to maintain FPS
                time.sleep(self.frame_time)
                
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def read(self):
        """
        Read the next frame from the camera simulator.
        
        Returns:
            (success, frame, metadata) tuple:
            - success: True if a frame was successfully read
            - frame: The image frame as a numpy array
            - metadata: Additional metadata about the frame
        """
        if not self.running:
            self.start()
            
        try:
            # Get image from buffer with timeout
            img, metadata = self.buffer.get(block=True, timeout=2.0)
            return True, img, metadata
            
        except queue.Empty:
            print("Warning: Camera buffer empty, waiting for frames...")
            return False, None, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None, None

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create camera simulator with 60 FPS
    cam = CameraSimulator(
        db_name="fruit_grading",
        collection_name="images",
        fps=60,
        random_order=True
    )
    
    # Start the camera
    cam.start()
    
    # Read and display frames
    try:
        plt.figure(figsize=(10, 8))
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:  # Process 100 frames
            success, frame, metadata = cam.read()
            
            if success:
                frame_count += 1
                
                # Clear previous image
                plt.clf()
                
                # Display the image with metadata
                plt.imshow(frame)
                title = f"Frame {frame_count}"
                if "category" in metadata:
                    title += f": {metadata['category']}"
                plt.title(title)
                plt.axis('off')
                
                # Update plot
                plt.pause(0.01)
                
                # Print metadata
                if frame_count % 10 == 0:
                    print(f"Frame {frame_count} metadata: {metadata}")
            else:
                # Wait briefly if no frame
                time.sleep(0.1)
        
        # Calculate actual FPS
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({actual_fps:.2f} FPS)")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop the camera
        cam.stop()
        plt.close()