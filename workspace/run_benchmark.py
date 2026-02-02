#!/usr/bin/env python3

import os
import subprocess
import time
import csv
import torch
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gc


RUN_ID = "test_001"

# rclone remote and container
RCLONE_REMOTE = "rclone_s3"  
# RCLONE_REMOTE = "chi_uc" 
RCLONE_CONTAINER = "ansh-lab4-tests-bucket"  # Your container name

# Mount point - use user's home directory to avoid permission issues
MOUNT_POINT = os.path.expanduser("~/rclone_mount")

# File paths
RESULTS_CSV = 'training_benchmark_results.csv'
GPU_CSV = 'gpu_utilization.csv'

RCLONE_OPTIONS = {
    # Cache settings
    'vfs_cache_mode': 'minimal',        # off, minimal, writes, full
    'vfs_cache_max_size': '20G',     # e.g., 5G, 10G, 20G, 50G
    'vfs_cache_max_age': '5h',       # e.g., 1h, 24h
    
    # Read performance
    'vfs_read_chunk_size': '256M',    # e.g., 16M, 64M, 128M, 256M
    'vfs_read_chunk_size_limit': 'off',  # e.g., 256M, 512M, off (unlimited)
    'vfs_read_ahead': '1G',        # e.g., 128M, 256M, 512M, 1G
    'buffer_size': '256M',           # e.g., 16M, 64M, 128M, 256M
    
    # Parallelism
    'transfers': '16',               # e.g., 4, 8, 16, 32
    'checkers': '8',                 # e.g., 4, 8, 16
    
    # Directory caching
    'dir_cache_time': '30m',         # e.g., 5m, 30m, 1h
    'attr_timeout': '30s',           # e.g., 1s, 10s, 30s, 1m
    
    # Stability/Retry settings
    'low_level_retries': '10',
    'retries': '3',
    'contimeout': '30s',
    'timeout': '120s',
}

DATALOADER_OPTIONS = {
    'batch_size': 64,
    'num_workers': 8,
    # 'prefetch_factor': 4,
    # 'pin_memory': True,
}

def setup_logging():
    log_dir = "logs/benchmark_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{RUN_ID}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting benchmark run: {RUN_ID}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def cleanup_existing_processes(logger=None):
    """Kill any existing rclone processes and unmount"""
    # Kill any existing rclone processes
    if logger:
        logger.info("Cleaning up existing rclone processes...")
    
    try:
        # Kill rclone processes
        kill_result = subprocess.run("pkill -f 'rclone mount'", shell=True, capture_output=True)
        if kill_result.returncode == 0:
            if logger:
                logger.info("Killed existing rclone processes")
        
        # Wait a bit for processes to die
        time.sleep(2)
        
        # Force unmount the mount point
        unmount_result = subprocess.run(f"sudo fusermount -uz {MOUNT_POINT}", shell=True, capture_output=True)
        if unmount_result.returncode == 0:
            if logger:
                logger.info(f"Force unmounted {MOUNT_POINT}")
        
        # Also try regular unmount
        unmount_result2 = subprocess.run(f"fusermount -u {MOUNT_POINT}", shell=True, capture_output=True)
        
        # Wait a bit more
        time.sleep(1)
        
    except Exception as e:
        if logger:
            logger.warning(f"Cleanup warning: {e}")

def build_mount_command(remote, container, mount_point, options):
    cmd = f"rclone mount {remote}:{container} {mount_point}"
    
    # Add options - remove --allow-other since it requires root or fuse group membership
    cmd += " --read-only"
    
    option_map = {
        'vfs_cache_mode': '--vfs-cache-mode',
        'vfs_cache_max_size': '--vfs-cache-max-size',
        'vfs_cache_max_age': '--vfs-cache-max-age',
        'vfs_read_chunk_size': '--vfs-read-chunk-size',
        'vfs_read_chunk_size_limit': '--vfs-read-chunk-size-limit',
        'vfs_read_ahead': '--vfs-read-ahead',
        'buffer_size': '--buffer-size',
        'transfers': '--transfers',
        'checkers': '--checkers',
        'dir_cache_time': '--dir-cache-time',
        'attr_timeout': '--attr-timeout',
        'low_level_retries': '--low-level-retries',
        'retries': '--retries',
        'contimeout': '--contimeout',
        'timeout': '--timeout',
    }
    
    for key, flag in option_map.items():
        value = options.get(key)
        if value:
            cmd += f" {flag} {value}"
    
    cmd += " --daemon"
    return cmd

def run_training_benchmark(data_dir, batch_size, num_workers, epochs=3, 
                          prefetch_factor=2, pin_memory=True, use_gpu=False, logger=None):
    if logger:
        logger.info("Setting up training benchmark...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num workers: {num_workers}")
        logger.info(f"  Prefetch factor: {prefetch_factor}")
        logger.info(f"  Pin memory: {pin_memory}")
        logger.info(f"  Epochs: {epochs}")
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"  Device: {device}")
    
    # Define transforms - aligned with train.py
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'training'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'validation'), transform=val_test_transform)
    eval_dataset = ImageFolder(os.path.join(data_dir, 'evaluation'), transform=val_test_transform)
    
    if logger:
        logger.info(f"Dataset sizes:")
        logger.info(f"  Training: {len(train_dataset)} images")
        logger.info(f"  Validation: {len(val_dataset)} images")
        logger.info(f"  Evaluation (test): {len(eval_dataset)} images")
        logger.info(f"  Classes: {len(train_dataset.classes)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create MobileNetV2 model - aligned with train.py architecture
    model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
    
    # Modify classifier for 11 classes (same as train.py)
    num_ftrs = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 11)
    )
    model = model.to(device)
    
    # For benchmark, freeze backbone features (similar to train.py approach)
    for param in model.features.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Training setup - aligned with train.py
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)  # 1e-4 as in train.py
    
    # Training metrics tracking
    training_start_time = time.perf_counter()
    epoch_times = []
    data_loading_times = []
    batch_processing_times = []
    training_losses = []
    validation_accuracies = []
    
    best_val_loss = float('inf')
    
    if logger:
        logger.info("Starting training...")
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_data_time = 0.0
        epoch_compute_time = 0.0
        
        if logger:
            logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        for i, (images, labels) in enumerate(train_loader):
            data_start = time.perf_counter()
            
            # Move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_gpu:
                torch.cuda.synchronize()
            
            data_time = time.perf_counter() - data_start
            epoch_data_time += data_time
            
            # Training step
            compute_start = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if use_gpu:
                torch.cuda.synchronize()
            
            compute_time = time.perf_counter() - compute_start
            epoch_compute_time += compute_time
            
            # Statistics - following train.py pattern
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress update every 50 batches
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / (i + 1)
                train_acc = 100. * correct / total
                data_throughput = total / (epoch_data_time + 1e-6)
                if logger:
                    logger.info(f"  Batch {i + 1:4d}: Loss={avg_loss:.4f}, "
                              f"Train Acc={train_acc:.2f}%, "
                              f"Data throughput={data_throughput:.1f} samples/sec")
        
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        data_loading_times.append(epoch_data_time)
        batch_processing_times.append(epoch_compute_time)
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        training_losses.append(train_loss)
        
        # Validation - following train.py validation function pattern
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        validation_accuracies.append(val_accuracy)
        
        if logger:
            logger.info(f"  Epoch {epoch + 1} Results:")
            logger.info(f"    Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
            logger.info(f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            logger.info(f"    Epoch Time: {epoch_time:.2f}s")
            logger.info(f"    Data Loading Time: {epoch_data_time:.2f}s ({100*epoch_data_time/epoch_time:.1f}%)")
            logger.info(f"    Compute Time: {epoch_compute_time:.2f}s ({100*epoch_compute_time/epoch_time:.1f}%)")
        
        # Save best model (following train.py pattern)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if logger:
                logger.info("    Validation loss improved. Best model updated.")
    
    total_training_time = time.perf_counter() - training_start_time
    
    # Final evaluation on test set
    if logger:
        logger.info("Final evaluation on test set...")
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    test_start_time = time.perf_counter()
    
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_time = time.perf_counter() - test_start_time
    
    # Calculate test metrics
    test_loss = test_running_loss / len(eval_loader)
    test_accuracy = 100. * test_correct / test_total
    
    # Additional metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    if logger:
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Compile results
    results = {
        # Training metrics
        'total_training_time_sec': total_training_time,
        'epochs': epochs,
        'final_training_loss': training_losses[-1],
        'best_val_accuracy': max(validation_accuracies),
        'final_val_accuracy': validation_accuracies[-1],
        'best_val_loss': best_val_loss,
        
        # Test metrics
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': precision * 100,
        'test_recall': recall * 100,
        'test_f1': f1 * 100,
        'test_inference_time_sec': test_time,
        
        # Data loading performance
        'avg_epoch_time_sec': np.mean(epoch_times),
        'avg_data_loading_time_sec': np.mean(data_loading_times),
        'avg_compute_time_sec': np.mean(batch_processing_times),
        'data_loading_ratio': np.mean(data_loading_times) / np.mean(epoch_times),
        'samples_per_sec_training': len(train_dataset) * epochs / total_training_time,
        'total_samples_processed': len(train_dataset) * epochs,
        
        # System metrics
        'batch_size': batch_size,
        'num_workers': num_workers,
        'total_train_samples': len(train_dataset),
        'total_val_samples': len(val_dataset),
        'total_test_samples': len(eval_dataset),
        'trainable_parameters': trainable_params,
    }
    
    # Cleanup
    del model, train_loader, val_loader, eval_loader
    del train_dataset, val_dataset, eval_dataset
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def save_results_to_csv(results, logger=None):
    """Save results to CSV file"""
    results_file = RESULTS_CSV
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'run_id', 'total_training_time_sec', 'epochs', 
            'final_training_loss', 'best_val_accuracy', 'final_val_accuracy', 'best_val_loss',
            'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_inference_time_sec',
            'avg_epoch_time_sec', 'avg_data_loading_time_sec', 'avg_compute_time_sec', 
            'data_loading_ratio', 'samples_per_sec_training', 'total_samples_processed',
            'batch_size', 'num_workers', 'total_train_samples', 'total_val_samples', 'total_test_samples', 
            'trainable_parameters', 'vfs_cache_mode', 'vfs_cache_max_size', 'vfs_read_chunk_size', 
            'buffer_size', 'transfers'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Prepare row data
        row_data = {
            'timestamp': results['timestamp'],
            'run_id': results['run_id'],
            'total_training_time_sec': results['total_training_time_sec'],
            'epochs': results['epochs'],
            'final_training_loss': results['final_training_loss'],
            'best_val_accuracy': results['best_val_accuracy'],
            'final_val_accuracy': results['final_val_accuracy'],
            'best_val_loss': results['best_val_loss'],
            'test_loss': results['test_loss'],
            'test_accuracy': results['test_accuracy'],
            'test_precision': results['test_precision'],
            'test_recall': results['test_recall'],
            'test_f1': results['test_f1'],
            'test_inference_time_sec': results['test_inference_time_sec'],
            'avg_epoch_time_sec': results['avg_epoch_time_sec'],
            'avg_data_loading_time_sec': results['avg_data_loading_time_sec'],
            'avg_compute_time_sec': results['avg_compute_time_sec'],
            'data_loading_ratio': results['data_loading_ratio'],
            'samples_per_sec_training': results['samples_per_sec_training'],
            'total_samples_processed': results['total_samples_processed'],
            'batch_size': results['batch_size'],
            'num_workers': results['num_workers'],
            'total_train_samples': results['total_train_samples'],
            'total_val_samples': results['total_val_samples'],
            'total_test_samples': results['total_test_samples'],
            'trainable_parameters': results['trainable_parameters'],
            'vfs_cache_mode': RCLONE_OPTIONS.get('vfs_cache_mode', 'default'),
            'vfs_cache_max_size': RCLONE_OPTIONS.get('vfs_cache_max_size', 'default'),
            'vfs_read_chunk_size': RCLONE_OPTIONS.get('vfs_read_chunk_size', 'default'),
            'buffer_size': RCLONE_OPTIONS.get('buffer_size', 'default'),
            'transfers': RCLONE_OPTIONS.get('transfers', 'default'),
        }
        
        writer.writerow(row_data)
    
    if logger:
        logger.info(f"Results appended to {results_file}")

def main():
    """Main function to run the complete benchmark"""
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("Configuration:")
        logger.info(f"  Run ID: {RUN_ID}")
        logger.info(f"  Remote: {RCLONE_REMOTE}:{RCLONE_CONTAINER}")
        logger.info(f"  Cache Mode: {RCLONE_OPTIONS.get('vfs_cache_mode', 'default')}")
        logger.info(f"  Batch Size: {DATALOADER_OPTIONS.get('batch_size', 64)}")
        logger.info(f"  Num Workers: {DATALOADER_OPTIONS.get('num_workers', 8)}")
        
        # Build mount command
        mount_cmd = build_mount_command(RCLONE_REMOTE, RCLONE_CONTAINER, MOUNT_POINT, RCLONE_OPTIONS)
        logger.info("Mount command:")
        logger.info(mount_cmd)
        
        # Cleanup any existing processes and mounts
        cleanup_existing_processes(logger)
        
        # Create mount point if it doesn't exist
        try:
            os.makedirs(MOUNT_POINT, exist_ok=True)
            logger.info(f"Mount point created/verified: {MOUNT_POINT}")
        except Exception as e:
            logger.error(f"Failed to create mount point {MOUNT_POINT}: {e}")
            return 1
        
        # Mount with new configuration
        logger.info(f"Mounting {RCLONE_REMOTE}:{RCLONE_CONTAINER} to {MOUNT_POINT}...")
        result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Mount failed!")
            logger.error(f"stderr: {result.stderr}")
            return 1
        else:
            # Wait for mount to be ready
            time.sleep(3)
            
            # Verify mount
            if os.path.exists(MOUNT_POINT) and os.listdir(MOUNT_POINT):
                logger.info(f"Mounted successfully!")
                logger.info(f"Contents: {os.listdir(MOUNT_POINT)}")
            else:
                logger.warning("Mount point exists but appears empty")
        
        # Check data directory
        DATA_DIR = MOUNT_POINT
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info(f"Contents: {os.listdir(DATA_DIR)}")
        
        # Run training benchmark
        logger.info("="*60)
        logger.info(f"TRAINING BENCHMARK RUN: {RUN_ID}")
        logger.info("="*60)
        
        results = run_training_benchmark(
            data_dir=DATA_DIR,
            batch_size=DATALOADER_OPTIONS.get('batch_size', 64),
            num_workers=DATALOADER_OPTIONS.get('num_workers', 8),
            epochs=10,  # Short benchmark run
            prefetch_factor=DATALOADER_OPTIONS.get('prefetch_factor', 2),
            pin_memory=DATALOADER_OPTIONS.get('pin_memory', False),
            use_gpu=True if device == 'cuda' else False,
            logger=logger
        )
        
        # Log summary results
        logger.info("="*60)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Training Performance:")
        logger.info(f"  Total training time: {results['total_training_time_sec']:.2f}s")
        logger.info(f"  Average epoch time: {results['avg_epoch_time_sec']:.2f}s")
        logger.info(f"  Data loading ratio: {results['data_loading_ratio']:.1%}")
        logger.info(f"  Training throughput: {results['samples_per_sec_training']:.1f} samples/sec")
        
        logger.info(f"Model Performance:")
        logger.info(f"  Best validation accuracy: {results['best_val_accuracy']:.2f}%")
        logger.info(f"  Final test accuracy: {results['test_accuracy']:.2f}%")
        logger.info(f"  Test F1 score: {results['test_f1']:.2f}%")
        
        logger.info(f"System Configuration:")
        logger.info(f"  Batch size: {results['batch_size']}")
        logger.info(f"  Number of workers: {results['num_workers']}")
        logger.info(f"  Trainable parameters: {results['trainable_parameters']:,}")
        
        # Save detailed results
        results['timestamp'] = datetime.now().isoformat()
        results['run_id'] = RUN_ID
        results['rclone_config'] = RCLONE_OPTIONS
        
        save_results_to_csv(results, logger)
        logger.info(f"Results saved with Run ID: {RUN_ID}")
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        return 1
    
    finally:
        # Cleanup - unmount and kill processes
        logger.info(f"Cleaning up...")
        
        # Kill any rclone processes
        subprocess.run("pkill -f 'rclone mount'", shell=True, capture_output=True)
        
        # Unmount
        logger.info(f"Unmounting {MOUNT_POINT}...")
        result = subprocess.run(f"fusermount -u {MOUNT_POINT}", shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Unmounted successfully!")
        else:
            # Try force unmount
            subprocess.run(f"sudo fusermount -uz {MOUNT_POINT}", shell=True, capture_output=True)
            logger.warning(f"Used force unmount")
        
        logger.info("="*60)
        logger.info("Benchmark completed!")
        logger.info("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())