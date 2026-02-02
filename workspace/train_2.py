import numpy as np
import os
import subprocess
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support
import gc

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

### Prepare data loaders

# Get data directory from environment variable, if set
# otherwise, assume data is in a directory named "Food-11"
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "./Food-11/Food-11")

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
train_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)

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
def train(model, dataloader, criterion, optimizer, device, epoch_num=0, logger_func=print):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_data_time = 0.0
    epoch_compute_time = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        data_start = time.perf_counter()
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Synchronize for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        data_time = time.perf_counter() - data_start
        epoch_data_time += data_time
        
        # Training step timing
        compute_start = time.perf_counter()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Synchronize for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        compute_time = time.perf_counter() - compute_start
        epoch_compute_time += compute_time

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress update every 50 batches
        if (i + 1) % 50 == 0:
            avg_loss = running_loss / (i + 1)
            train_acc = 100. * correct / total
            data_throughput = total / (epoch_data_time + 1e-6)
            logger_func(f"  Batch {i + 1:4d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Data throughput={data_throughput:.1f} samples/sec")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, epoch_data_time, epoch_compute_time

# validate function - one epoch
def validate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_predictions, all_labels
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

# Initial training: only the classification head, freeze the backbone/base model
for param in food11_model.features.parameters():
    param.requires_grad = False

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(food11_model.classifier.parameters(), lr=config.get("lr", 1e-4))

### Training loop for initial training

# Training metrics tracking
training_start_time = time.perf_counter()
epoch_times = []
data_loading_times = []
batch_processing_times = []
training_losses = []
validation_accuracies = []

best_val_loss = float('inf')

print(f"Dataset sizes:")
print(f"  Training: {len(train_dataset)} images")
print(f"  Validation: {len(val_dataset)} images") 
print(f"  Test: {len(test_dataset)} images")
print(f"  Classes: {len(train_dataset.classes)}")
print(f"  Trainable parameters: {trainable_params:,}")
print("Starting initial training...")

# train new classification head on pre-trained model for a few epochs
for epoch in range(config.get("initial_epochs", 5)):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch+1}/{config.get('initial_epochs', 5)}")
    
    train_loss, train_acc, epoch_data_time, epoch_compute_time = train(
        food11_model, train_loader, criterion, optimizer, device, epoch, print
    )
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    data_loading_times.append(epoch_data_time)
    batch_processing_times.append(epoch_compute_time)
    training_losses.append(train_loss)
    validation_accuracies.append(val_acc * 100)
    
    # Detailed logging
    print(f"  Epoch {epoch+1} Results:")
    print(f"    Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%")
    print(f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
    print(f"    Epoch Time: {epoch_time:.2f}s")
    print(f"    Data Loading Time: {epoch_data_time:.2f}s ({100*epoch_data_time/epoch_time:.1f}%)")
    print(f"    Compute Time: {epoch_compute_time:.2f}s ({100*epoch_compute_time/epoch_time:.1f}%)")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(food11_model.state_dict(), "food11.pth")
        print("    Validation loss improved. Model saved.")

### Un-freeze backbone/base model and keep training with smaller learning rate

# unfreeze to fine-tune the entire model
for param in food11_model.features.parameters():
    param.requires_grad = True

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

# optimizer for the entire model with a smaller learning rate for fine-tuning
optimizer = optim.Adam(food11_model.parameters(), lr=config.get("fine_tune_lr", 1e-5))

patience_counter = 0

# Fine-tune entire model for the remaining epochs
for epoch in range(config.get("initial_epochs", 5), config.get("total_epochs", 20)):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch+1}/{config.get('total_epochs', 20)} (Fine-tuning)")
    
    train_loss, train_acc, epoch_data_time, epoch_compute_time = train(
        food11_model, train_loader, criterion, optimizer, device, epoch, print
    )
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    data_loading_times.append(epoch_data_time)
    batch_processing_times.append(epoch_compute_time)
    training_losses.append(train_loss)
    validation_accuracies.append(val_acc * 100)
    
    # Detailed logging
    print(f"  Epoch {epoch+1} Results:")
    print(f"    Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%")
    print(f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
    print(f"    Epoch Time: {epoch_time:.2f}s")
    print(f"    Data Loading Time: {epoch_data_time:.2f}s ({100*epoch_data_time/epoch_time:.1f}%)")
    print(f"    Compute Time: {epoch_compute_time:.2f}s ({100*epoch_compute_time/epoch_time:.1f}%)")

    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(food11_model.state_dict(), "food11.pth")
        print("    Validation loss improved. Model saved.")

    else:
        patience_counter += 1
        print(f"    No improvement in validation loss. Patience counter: {patience_counter}")

    if patience_counter >= config.get("patience", 5):
        print("    Early stopping triggered.")
        break


### Evaluate on test set
print("\n" + "="*60)
print("FINAL TEST EVALUATION")
print("="*60)

total_training_time = time.perf_counter() - training_start_time
test_start_time = time.perf_counter()

test_loss, test_acc, all_predictions, all_labels = validate(
    food11_model, test_loader, criterion, device, return_predictions=True
)

test_time = time.perf_counter() - test_start_time

# Calculate additional metrics using sklearn
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

# Calculate performance metrics
avg_epoch_time = np.mean(epoch_times)
avg_data_loading_time = np.mean(data_loading_times)
avg_compute_time = np.mean(batch_processing_times)
data_loading_ratio = avg_data_loading_time / avg_epoch_time
samples_per_sec_training = len(train_dataset) * len(epoch_times) / total_training_time

print(f"\nTraining Performance:")
print(f"  Total training time: {total_training_time:.2f}s")
print(f"  Average epoch time: {avg_epoch_time:.2f}s")
print(f"  Data loading ratio: {data_loading_ratio:.1%}")
print(f"  Training throughput: {samples_per_sec_training:.1f} samples/sec")

print(f"\nModel Performance:")
print(f"  Best validation accuracy: {max(validation_accuracies):.2f}%")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Test Precision: {precision*100:.2f}%")
print(f"  Test Recall: {recall*100:.2f}%")
print(f"  Test F1 Score: {f1*100:.2f}%")
print(f"  Test inference time: {test_time:.2f}s")

print(f"\nSystem Configuration:")
print(f"  Batch size: {config.get('batch_size', 64)}")
print(f"  Number of workers: {config.get('num_workers', 8)}")
print(f"  Trainable parameters: {trainable_params:,}")

# Cleanup
del food11_model, train_loader, val_loader, test_loader
del train_dataset, val_dataset, test_dataset
if device.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()

print("="*60)
print("Training completed!")
print("="*60)