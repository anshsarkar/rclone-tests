import numpy as np
import os
import subprocess
import time
import logging
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
import boto3
import io

### Configure the training job 
# All hyperparameters will be set here, in one convenient place
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "num_workers": 8,
    "prefetch_factor": 2,
    "pin_memory": False,
    "lr": 1e-4,
    "fine_tune_lr": 1e-5,
    "model_architecture": "MobileNetV2",
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2, 
    "color_jitter_hue": 0.1
}

# Setup logging
def setup_logging():
    log_dir = "logs/torch_s3"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_s3_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("PYTORCH DIRECT S3 TRAINING")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    return logger

logger = setup_logging()

# Start timing the entire script
script_start_time = time.time()
logger.info(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

### Writing custom data loader to load from s3

class S3ImageFolder(torch.utils.data.Dataset):
    def __init__(self, bucket_name, root_prefix, transform=None):
        super().__init__()

        self.bucket_name = bucket_name
        self.root_prefix = root_prefix
        self.transform = transform

        self.s3_client = boto3.client(
            's3', 
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        )

        self.image_paths = []
        self.labels = set()

        print(f"Listing objects in s3://{bucket_name}/{root_prefix}...")
        logger.info(f"Loading data directly from S3: s3://{bucket_name}/{root_prefix}")
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=root_prefix)

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = key[len(root_prefix):].lstrip('/')
                    parts = relative_path.split('/')
                    if len(parts) > 1:
                        label = parts[0]
                        self.labels.add(label)
                        self.image_paths.append((key, label))

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels))}
        print(f"Found {len(self.image_paths)} images across {len(self.labels)} classes.")
        logger.info(f"Dataset loaded from S3: {len(self.image_paths)} images, {len(self.labels)} classes")
        
        # Initialize counter for S3 access logging
        self.s3_access_count = 0

    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        if not hasattr(self, 's3_client'):
            self.s3_client = boto3.client(
            's3', 
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        )
        
        key, label = self.image_paths[idx]
        label = self.label_to_idx[label]

        # Log S3 access for first few files and then periodically
        self.s3_access_count += 1
        if self.s3_access_count <= 5 or self.s3_access_count % 1000 == 0:
            logger.info(f"Reading image directly from S3: s3://{self.bucket_name}/{key} (access #{self.s3_access_count})")

        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        img_data = obj['Body'].read()

        image = Image.open(io.BytesIO(img_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

### Prepare data loaders

# Get data directory from environment variable, if set
# otherwise, assume data is in a directory named "Food-11"
# food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "./rclone_mount")

s3_bucket = os.getenv("S3_BUCKET_NAME", "ansh-lab4-tests-bucket")

logger.info("="*60)
logger.info("DATA LOADING CONFIGURATION")
logger.info("="*60)
logger.info(f"S3 Bucket: {s3_bucket}")
logger.info(f"S3 Endpoint: {os.getenv('S3_ENDPOINT_URL', 'Not set')}")
logger.info("Data loading: DIRECT FROM S3 (no filesystem mount)")
logger.info("="*60)

# Define transforms for training data augmentation
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=config.get("random_horizontal_flip", 0.5)),
    transforms.RandomRotation(config.get("random_rotation", 15)),
    transforms.ColorJitter(
        brightness=config.get("color_jitter_brightness", 0.2),
        contrast=config.get("color_jitter_contrast", 0.2),
        saturation=config.get("color_jitter_saturation", 0.2),
        hue=config.get("color_jitter_hue", 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create data loaders
# train_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=train_transform)
# val_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
# test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)

train_dataset = S3ImageFolder(bucket_name=s3_bucket, root_prefix='training', transform=train_transform)
val_dataset = S3ImageFolder(bucket_name=s3_bucket, root_prefix='validation', transform=val_test_transform)
test_dataset = S3ImageFolder(bucket_name=s3_bucket, root_prefix='evaluation', transform=val_test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.get("batch_size", 64),
    shuffle=True,
    num_workers=config.get("num_workers", 8),
    pin_memory=config.get("pin_memory", True),
    prefetch_factor=config.get("prefetch_factor", 4),
    persistent_workers=True if config.get("num_workers", 8) > 0 else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.get("batch_size", 64),
    shuffle=False,
    num_workers=config.get("num_workers", 8),
    pin_memory=config.get("pin_memory", True),
    prefetch_factor=config.get("prefetch_factor", 4),
    persistent_workers=True if config.get("num_workers", 8) > 0 else False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.get("batch_size", 64),
    shuffle=False,
    num_workers=config.get("num_workers", 8),
    pin_memory=config.get("pin_memory", True),
    prefetch_factor=config.get("prefetch_factor", 4),
    persistent_workers=True if config.get("num_workers", 8) > 0 else False
)

### Define training and validation/test functions
# This is Pytorch boilerplate

# training function - one epoch
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# validate function - one epoch
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc =  correct / total

    return epoch_loss, epoch_acc

### Define the model


# Define model
food11_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
num_ftrs = food11_model.last_channel
food11_model.classifier = nn.Sequential(
    nn.Dropout(config.get("dropout_probability", 0.5)),
    nn.Linear(num_ftrs, 11)
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food11_model = food11_model.to(device)

logger.info("="*60)
logger.info("MODEL CONFIGURATION")
logger.info("="*60)
logger.info(f"Model: {config.get('model_architecture', 'MobileNetV2')}")
logger.info(f"Device: {device}")
logger.info(f"Batch size: {config.get('batch_size', 64)}")
logger.info(f"Num workers: {config.get('num_workers', 8)}")
logger.info("="*60)

# Initial training: only the classification head, freeze the backbone/base model
for param in food11_model.features.parameters():
    param.requires_grad = False

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

logger.info("INITIAL TRAINING PHASE - Classification head only")
logger.info(f"Trainable parameters: {trainable_params:,}")
logger.info(f"Initial epochs: {config.get('initial_epochs', 5)}")
logger.info(f"Learning rate: {config.get('lr', 1e-4)}")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(food11_model.classifier.parameters(), lr=config.get("lr", 1e-4))

### Training loop for initial training

best_val_loss = float('inf')

# train new classification head on pre-trained model for a few epochs
for epoch in range(config.get("initial_epochs", 5)):
    epoch_start_time = time.time()
    train_loss, train_acc = train(food11_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Time: {epoch_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(food11_model.state_dict(), "food11.pth")
        print("  Validation loss improved. Model saved.")

### Un-freeze backbone/base model and keep training with smaller learning rate

# unfreeze to fine-tune the entire model
for param in food11_model.features.parameters():
    param.requires_grad = True

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

logger.info("FINE-TUNING PHASE - Entire model")
logger.info(f"Trainable parameters: {trainable_params:,}")
logger.info(f"Fine-tune learning rate: {config.get('fine_tune_lr', 1e-5)}")
logger.info(f"Total epochs: {config.get('total_epochs', 20)}")

# optimizer for the entire model with a smaller learning rate for fine-tuning
optimizer = optim.Adam(food11_model.parameters(), lr=config.get("fine_tune_lr", 1e-5))

patience_counter = 0

# Fine-tune entire model for the remaining epochs
for epoch in range(config.get("initial_epochs", 5), config.get("total_epochs", 20)):

    epoch_start_time = time.time()
    train_loss, train_acc = train(food11_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Time: {epoch_time:.2f}s")

    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(food11_model.state_dict(), "food11.pth")
        print("  Validation loss improved. Model saved.")

    else:
        patience_counter += 1
        print(f"  No improvement in validation loss. Patience counter: {patience_counter}")

    if patience_counter >= config.get("patience", 5):
        print("  Early stopping triggered.")
        break


### Evaluate on test set
test_loss, test_acc = validate(food11_model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Calculate total script execution time
script_end_time = time.time()
total_execution_time = script_end_time - script_start_time
execution_minutes = total_execution_time / 60

logger.info("="*60)
logger.info("FINAL RESULTS")
logger.info("="*60)
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_acc:.2f}%")
logger.info(f"Total execution time: {total_execution_time:.2f} seconds ({execution_minutes:.2f} minutes)")
logger.info(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("Training completed - Data loaded directly from S3")
logger.info("="*60)