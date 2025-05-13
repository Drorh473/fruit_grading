import torch
import cv2
from preprocessing_from_camera import custom_preprocessing
from env_config import get_model_config

class FeatureMapExtractor:
    def __init__(self, model):
        """
        Initialize the Feature Map Extractor with a trained model
        
        Args:
            model: A trained PyTorch model
        """
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store the model
        self.model = model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Register hooks to extract feature maps
        self.feature_maps = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to extract feature maps from different layers"""
        # Function to capture outputs
        def get_features(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        self.hooks.append(self.model.conv1.register_forward_hook(get_features('conv1')))
        self.hooks.append(self.model.stage2.register_forward_hook(get_features('stage2')))
        self.hooks.append(self.model.stage3.register_forward_hook(get_features('stage3')))
        self.hooks.append(self.model.stage4.register_forward_hook(get_features('stage4')))
        self.hooks.append(self.model.conv5.register_forward_hook(get_features('conv5')))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def preprocess_frame(self, frame):
        """Preprocess a camera frame for model input"""
        if frame is None:
            raise ValueError("Received empty frame from camera")
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply custom preprocessing
        processed_frame = custom_preprocessing(frame_rgb)
        
        # Convert to PyTorch tensor and add batch dimension
        frame_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        return frame_tensor
    
    def extract_feature_maps(self, frame):
        """
        Run model and extract feature maps from a camera frame
        
        Args:
            frame: Camera frame as numpy array
        
        Returns:
            Dictionary with feature maps
        """
        # Preprocess frame
        frame_tensor = self.preprocess_frame(frame)
        
        # Clear previous feature maps
        self.feature_maps = {}
        
        # Run model with no gradient computation
        with torch.no_grad():
            # Forward pass
            _ = self.model(frame_tensor)
            
            # Get feature maps
            feature_maps = {name: feat.cpu().numpy() for name, feat in self.feature_maps.items()}
            
            return feature_maps


def get_feature_maps_from_camera(model, frame):
    """
    Get feature maps from a camera frame using the provided model
    
    Args:
        model: A trained PyTorch model
        frame: Camera frame as numpy array
        
    Returns:
        Dictionary with feature maps
    """
    # Create feature map extractor with the provided model
    extractor = FeatureMapExtractor(model)
    
    try:
        # Extract feature maps
        feature_maps = extractor.extract_feature_maps(frame)
        return feature_maps
    finally:
        # Clean up hooks to prevent memory leaks
        extractor.remove_hooks()
