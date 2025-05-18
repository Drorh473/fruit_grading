import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torchvision.models as models
from torchvision.models import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights
from torchvision.models import ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights
import torchvision.transforms as transforms

from preprocessing.preprocessing_from_db import preprocessing_from_db
from env_config import get_model_config, get_dataset_paths

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_cnn_from_db(
    num_epochs=10,
    cycle_length=2,  # Switch optimizer every 5 epochs
    adam_lr=0.001,
    sgd_lr=0.01,
    model_save_dir=None,
    model_name="shufflenet_v2_fruit_grader",
    model_variant=None
):
    """
    Train ShuffleNet V2 on the fruit dataset using data from MongoDB
    
    Args:
        num_epochs: Number of training epochs
        cycle_length: Number of epochs before switching optimizer
        adam_lr: Learning rate for Adam optimizer
        sgd_lr: Learning rate for SGD optimizer
        model_save_dir: Directory to save models
        model_name: Base name for the saved model
        model_variant: ShuffleNet V2 variant ('0.5x', '1.0x', '1.5x', '2.0x')
    
    Returns:
        Trained model and training statistics
    """
    # Get configuration from environment
    model_config = get_model_config()
    
    # Use environment config values if not provided as parameters
    if model_save_dir is None:
        model_save_dir = model_config.get('model_dir', 'saved_models')
        
    if model_variant is None:
        model_variant = model_config.get('default_variant', '1.0x')
    
    print("\n==== Starting ShuffleNet V2 Training ====\n")
    print(f"Using model variant: {model_variant}")
    print(f"Models will be saved to: {model_save_dir}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 1. Load dataset
    print("Loading and preparing dataset...")
    train_gen, val_gen, test_gen, class_indices, counts = preprocessing_from_db()
    
    if train_gen is None:
        raise ValueError("Failed to load training dataset")
    
    img_size = (224, 224)  # Use fixed size since images are resized in preprocessing_from_db
    
    num_classes = len(class_indices)
    print(f"Training model for {num_classes} classes: {list(class_indices.keys())}")
    print(f"Training samples: {counts['training']}, Validation samples: {counts['validation']}, Testing samples: {counts['testing']}")
    
    # 2. Initialize model
    print(f"Initializing ShuffleNet V2 ({model_variant})...")
    
    # Use weights parameter instead of pretrained for each model variant
    if model_variant == "0.5x":
        model = models.shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    elif model_variant == "1.0x":
        model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    elif model_variant == "1.5x":
        model = models.shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
    elif model_variant == "2.0x":
        model = models.shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    
    # Modify final layer for our classification task
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Set up cross entropy loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up both optimizers as used in the article
    adam_optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    sgd_optimizer = optim.SGD(model.parameters(), lr=sgd_lr, momentum=0.9, weight_decay=0.0001)
    
    # Track which optimizer is currently being used
    current_optimizer_name = "Adam"  # Start with Adam
    
    # Set up schedulers
    # Calculate approximate steps per epoch based on sample count and batch size
    steps_per_epoch = train_gen.num_batches
    step_size_up = steps_per_epoch * 2  # 2 epochs per cycle
    
    adam_scheduler = CyclicLR(
        adam_optimizer,
        base_lr=adam_lr / 10,
        max_lr=adam_lr,
        step_size_up=step_size_up,
        mode="triangular",
        cycle_momentum=False
    )
    
    sgd_scheduler = CyclicLR(
        sgd_optimizer,
        base_lr=sgd_lr / 10,
        max_lr=sgd_lr,
        step_size_up=step_size_up,
        mode="triangular",
        cycle_momentum=True
    )
    
    # 5. Training and validation
    print("\nStarting training...\n")
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'optimizer': []  # Track which optimizer was used in each epoch
    }
    
    # Track the best model and its stats
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_optimizer_states = {
        'adam': None,
        'sgd': None
    }
    best_scheduler_states = {
        'adam': None,
        'sgd': None
    }
    best_optimizer_name = None
    
    # Create ImageNet normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Determine which optimizer to use for this epoch
        current_cycle = (epoch // cycle_length) % 2
        if current_cycle == 0:
            optimizer = adam_optimizer
            scheduler = adam_scheduler
            current_optimizer_name = "Adam"
            print(f"Epoch {epoch+1}/{num_epochs}: Using Adam optimizer")
        else:
            optimizer = sgd_optimizer
            scheduler = sgd_scheduler
            current_optimizer_name = "SGD"
            print(f"Epoch {epoch+1}/{num_epochs}: Using SGD optimizer")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Get batches from our generator
        train_iterator = train_gen()
        for inputs, targets in tqdm(train_iterator, desc=f"Epoch {epoch+1}/{num_epochs} [{current_optimizer_name}]"):
            # Convert to float first and normalize to 0-1 range 
            inputs = torch.from_numpy(inputs).float().div(255.0).to(device)
            
            # Apply ImageNet normalization
            inputs = normalize(inputs.permute(0, 3, 1, 2))  # permute to get NCHW format
            
            # Convert targets to float
            targets = torch.from_numpy(targets).float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, target_indices = torch.max(targets, 1)
            train_total += targets.size(0)
            train_correct += (predicted == target_indices).sum().item()
        
        train_loss = train_loss / train_gen.num_batches
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Get batches from validation dataset
        val_iterator = val_gen()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_iterator, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Convert to float first and normalize to 0-1 range 
                inputs = torch.from_numpy(inputs).float().div(255.0).to(device)
                
                # Apply ImageNet normalization
                inputs = normalize(inputs.permute(0, 3, 1, 2))  # permute to get NCHW format
                
                # Convert targets to float
                targets = torch.from_numpy(targets).float().to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, target_indices = torch.max(targets, 1)
                val_total += targets.size(0)
                val_correct += (predicted == target_indices).sum().item()
        
        val_loss = val_loss / val_gen.num_batches
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['optimizer'].append(current_optimizer_name)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} [{current_optimizer_name}] - {epoch_time:.2f}s - "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.2f}%")
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_optimizer_states['adam'] = adam_optimizer.state_dict().copy()
            best_optimizer_states['sgd'] = sgd_optimizer.state_dict().copy()
            best_scheduler_states['adam'] = adam_scheduler.state_dict().copy()
            best_scheduler_states['sgd'] = sgd_scheduler.state_dict().copy()
            best_optimizer_name = current_optimizer_name
            print(f"New best model with validation accuracy: {val_acc:.2f}%")
    
    # Save the best model at the end of training
    print(f"\nTraining complete. Saving the best model from epoch {best_epoch} with accuracy {best_val_acc:.2f}%")
    
    # Set the model to the best state
    model.load_state_dict(best_model_state)
    
    # Save the best model
    best_model_path = os.path.join(
        model_save_dir,
        f"{model_name}_{model_variant}_best.pth"
    )
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'adam_optimizer_state_dict': best_optimizer_states['adam'],
        'sgd_optimizer_state_dict': best_optimizer_states['sgd'],
        'adam_scheduler_state_dict': best_scheduler_states['adam'],
        'sgd_scheduler_state_dict': best_scheduler_states['sgd'],
        'current_optimizer': best_optimizer_name,
        'val_acc': best_val_acc,
        'class_indices': class_indices,
        'num_classes': num_classes,
        'img_size': img_size,
        'model_variant': model_variant
    }, best_model_path)
    
    print(f"Best model saved at: {best_model_path}")
    
    # 6. Evaluate on test set
    print("\nEvaluating best model on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    # Get batches from test dataset
    test_iterator = test_gen()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_iterator, desc="Testing"):
            # Convert to float first and normalize to 0-1 range 
            inputs = torch.from_numpy(inputs).float().div(255.0).to(device)
            
            # Apply ImageNet normalization
            inputs = normalize(inputs.permute(0, 3, 1, 2))  # permute to get NCHW format
            
            # Convert targets to float
            targets = torch.from_numpy(targets).float().to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Track statistics
            _, predicted = torch.max(outputs, 1)
            _, target_indices = torch.max(targets, 1)
            test_total += targets.size(0)
            test_correct += (predicted == target_indices).sum().item()
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target_indices.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f"Test accuracy of best model: {test_acc:.2f}%")
    
    # 7. Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss with optimizer highlighting
    plt.subplot(2, 2, 3)
    for i in range(len(history['train_loss'])):
        color = 'blue' if history['optimizer'][i] == 'Adam' else 'red'
        if i > 0:
            plt.plot([i-1, i], [history['train_loss'][i-1], history['train_loss'][i]], 
                     color=color, linewidth=2)
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Loss by Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    
    # Add legend for optimizer colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Adam'),
        Line2D([0], [0], color='red', lw=2, label='SGD'),
        Line2D([0], [0], color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    ]
    plt.legend(handles=legend_elements)
    
    # Plot accuracy with optimizer highlighting
    plt.subplot(2, 2, 4)
    for i in range(len(history['val_acc'])):
        color = 'blue' if history['optimizer'][i] == 'Adam' else 'red'
        if i > 0:
            plt.plot([i-1, i], [history['val_acc'][i-1], history['val_acc'][i]], 
                     color=color, linewidth=2)
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Validation Accuracy by Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(model_save_dir, f"{model_name}_{model_variant}_training_curves.png"))
    plt.close()
    
    # Return the trained model and best model path
    return model, best_model_path, class_indices

if __name__ == "__main__":
    # Get configuration from environment
    model_config = get_model_config()
    
    # Train the model using environment configuration
    model, best_model_path, class_indices = train_cnn_from_db(
        num_epochs=10,  # Reduced from 30 for faster training
        adam_lr=0.001,
        sgd_lr=0.01,
        cycle_length=2,  # Switch optimizer every 2 epochs
        model_save_dir=model_config.get('model_dir', 'saved_models'),
        model_variant=model_config.get('default_variant', '1.0x')
    )
    
    print(f"\nTraining complete. Best model saved at: {best_model_path}")
    print(f"Class indices: {class_indices}")