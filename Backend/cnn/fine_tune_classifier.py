"""Fine-tuned ShuffleNetV2 for fruit classification."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

LABEL_MAPPING = {'market': 0, 'standard': 1, 'premium': 2}
REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
NUM_CLASSES = 3


class FruitDataset(Dataset):
    """Dataset for fruit images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class FineTunedShuffleNet(nn.Module):
    """ShuffleNetV2 with custom classifier head."""

    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()
        self.backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_last_stage(self):
        for name, param in self.backbone.named_parameters():
            if 'stage4' in name or 'conv5' in name or 'fc' in name:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_data_loaders(train_paths, train_labels, test_paths, test_labels, batch_size=8):
    """Create data loaders with augmentation for training."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FruitDataset(train_paths, train_labels, train_transform)
    test_dataset = FruitDataset(test_paths, test_labels, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(test_loader) if len(test_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    confidences = [all_probs[i, pred] for i, pred in enumerate(all_preds)]
    avg_confidence = np.mean(confidences) if confidences else 0

    return avg_loss, accuracy, avg_confidence, all_preds, all_labels


def train_fine_tuned_model(train_paths, train_labels, test_paths, test_labels,
                           epochs=50, learning_rate=0.001, batch_size=8,
                           early_stopping_patience=10, unfreeze_backbone=False):
    """Train fine-tuned ShuffleNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FineTunedShuffleNet(num_classes=NUM_CLASSES, freeze_backbone=True)
    if unfreeze_backbone:
        model.unfreeze_last_stage()
    model = model.to(device)

    train_loader, test_loader = create_data_loaders(
        train_paths, train_labels, test_paths, test_labels, batch_size
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_loss, val_acc, avg_conf, preds, labels = evaluate(model, test_loader, criterion, device)
    train_loss, train_acc, _, _, _ = evaluate(model, train_loader, criterion, device)

    results = {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'test_loss': val_loss,
        'test_accuracy': val_acc,
        'avg_confidence': avg_conf,
        'predictions': preds,
        'labels': labels,
        'final_epoch': len(history['train_loss'])
    }

    return model, history, results


def save_fine_tuned_model(model, save_path, metadata=None):
    """Save the fine-tuned model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': 'FineTunedShuffleNet',
        'num_classes': NUM_CLASSES,
        'label_mapping': LABEL_MAPPING,
    }

    if metadata:
        save_dict['metadata'] = metadata

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")


def load_fine_tuned_model(model_path, device=None):
    """Load a fine-tuned model for inference."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    model = FineTunedShuffleNet(num_classes=checkpoint.get('num_classes', NUM_CLASSES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get('label_mapping', LABEL_MAPPING)


# Cache for loaded model
_FINE_TUNED_CACHE = {
    'model': None,
    'label_mapping': None,
    'device': None
}


def get_cached_model(model_path):
    """Get cached model or load it."""
    if _FINE_TUNED_CACHE['model'] is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, label_mapping = load_fine_tuned_model(model_path, device)
        _FINE_TUNED_CACHE['model'] = model
        _FINE_TUNED_CACHE['label_mapping'] = label_mapping
        _FINE_TUNED_CACHE['device'] = device
        print(f"Fine-tuned model loaded on {device}")

    return _FINE_TUNED_CACHE['model'], _FINE_TUNED_CACHE['label_mapping'], _FINE_TUNED_CACHE['device']


def predict_single_image(image_path, model_path):
    """Predict class for a single image."""
    model, label_mapping, device = get_cached_model(model_path)
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    predicted_class = predicted.item()
    confidence_value = confidence.item()
    predicted_type = reverse_mapping.get(predicted_class, 'unknown')

    return predicted_type, confidence_value


def predict_multiple_images(image_paths, model_path):
    """
    Predict class for multiple images (e.g., multi-view fruit).
    Uses voting/averaging across all views.
    """
    model, label_mapping, device = get_cached_model(model_path)
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and transform all images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    if not images:
        return None, 0.0

    # Stack into batch
    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)

    # Average probabilities across all views
    avg_probs = probs.mean(dim=0)
    confidence, predicted = avg_probs.max(0)

    predicted_class = predicted.item()
    confidence_value = confidence.item()
    predicted_type = reverse_mapping.get(predicted_class, 'unknown')

    return predicted_type, confidence_value


def clear_model_cache():
    """Clear the cached model to free memory."""
    global _FINE_TUNED_CACHE
    _FINE_TUNED_CACHE = {'model': None, 'label_mapping': None, 'device': None}
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
