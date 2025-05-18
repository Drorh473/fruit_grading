import torch
import cv2
import numpy as np
from preprocessing.preprocessing_from_camera import custom_preprocessing
import torchvision.models as models

def register_hooks(model):

    # Dictionary to store the feature maps
    feature_maps = {}
    hooks = []
    
    # Function to capture outputs
    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks for key layers
    hooks.append(model.conv1.register_forward_hook(get_features('conv1')))
    hooks.append(model.stage2.register_forward_hook(get_features('stage2')))
    hooks.append(model.stage3.register_forward_hook(get_features('stage3')))
    hooks.append(model.stage4.register_forward_hook(get_features('stage4')))
    hooks.append(model.conv5.register_forward_hook(get_features('conv5')))
    
    return feature_maps, hooks

def remove_hooks(hooks):

    for hook in hooks:
        hook.remove()

def preprocess_frame(frame, device):
    """
    Preprocess a camera frame for model input.
    
    Args:
        frame: Camera frame as numpy array
        device: PyTorch device (CPU or GPU)
    
    Returns:
        PyTorch tensor ready for model input
    """
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
    # Need to transpose from HWC to CHW format for PyTorch
    frame_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return frame_tensor

def extract_feature_maps(model, frame):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Register hooks
    feature_maps_dict, hooks = register_hooks(model)
    
    try:
        # Preprocess frame
        frame_tensor = preprocess_frame(frame, device)
        
        # Run model with no gradient computation
        with torch.no_grad():
            # Forward pass
            _ = model(frame_tensor)
            
            # Transform feature maps to (h', w', c') format
            result_maps = {}
            for name, feat in feature_maps_dict.items():
                # Move to CPU as numpy array
                feat_np = feat.cpu().numpy()
                
                # Convert from (batch_size, c', h', w') to (h', w', c')
                # 1. Transpose dimensions to get (batch, h, w, c)
                feat_np = np.transpose(feat_np, (0, 2, 3, 1))
                
                # 2. Remove the batch dimension (index 0) since batch size is 1
                feat_np = feat_np[0]  # Now shape is (h', w', c')
                
                result_maps[name] = feat_np
            
            return result_maps
    finally:
        # Clean up hooks to prevent memory leaks
        remove_hooks(hooks)

def get_feature_maps_from_camera( frame):
    # Load pre-trained ShuffleNetV2 model
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.eval()  # Set to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Simply call the extract_feature_maps function
    return extract_feature_maps(model, frame)
