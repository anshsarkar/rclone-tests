#!/usr/bin/env python3

# Remember to run gpu_monitoring in a separate terminal while running this script:
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

# Mount point - use user's home directory to avoid permission issues
MOUNT_POINT = os.path.expanduser("~/rclone_mount")

# File paths
RESULTS_CSV = 'multiple_benchmark_results.csv'

# Define multiple benchmark configurations to test
# BENCHMARK_CONFIGURATIONS = [
#     {
#         'run_id': 'rclone_full_workers8',
#         'description': 'Full rclone cache, 8 workers',
#         'rclone_remote': 'rclone_s3',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '20G',
#             'vfs_cache_max_age': '5h',
#             'vfs_read_chunk_size': '256M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '2G',
#             'buffer_size': '512M',
#             'transfers': '32',
#             'checkers': '16',
#             'dir_cache_time': '1h',
#             'attr_timeout': '30s',
#             'low_level_retries': '10',
#             'retries': '3',
#             'contimeout': '30s',
#             'timeout': '120s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 8,
#             'prefetch_factor': 4,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_aggressive_workers12',
#         'description': 'Aggressive rclone settings, 12 workers',
#         'rclone_remote': 'rclone_s3',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '50G',
#             'vfs_cache_max_age': '8h',
#             'vfs_read_chunk_size': '512M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '4G',
#             'buffer_size': '1G',
#             'transfers': '48',
#             'checkers': '24',
#             'dir_cache_time': '2h',
#             'attr_timeout': '60s',
#             'low_level_retries': '15',
#             'retries': '5',
#             'contimeout': '60s',
#             'timeout': '300s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 12,
#             'prefetch_factor': 6,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_balanced_optimized',
#         'description': 'Balanced optimized configuration',
#         'rclone_remote': 'rclone_s3',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '35G',
#             'vfs_cache_max_age': '6h',
#             'vfs_read_chunk_size': '384M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '3G',
#             'buffer_size': '768M',
#             'transfers': '40',
#             'checkers': '20',
#             'dir_cache_time': '90m',
#             'attr_timeout': '45s',
#             'low_level_retries': '12',
#             'retries': '4',
#             'contimeout': '45s',
#             'timeout': '240s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 10,
#             'prefetch_factor': 5,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_balanced_streaming',
#         'description': 'Balanced optimized with streaming and no-modtime',
#         'rclone_remote': 'rclone_s3',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '35G',
#             'vfs_cache_max_age': '6h',
#             'vfs_read_chunk_size': '384M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '3G',
#             'vfs_read_chunk_streams': '32',
#             'no_modtime': True,
#             'buffer_size': '768M',
#             'transfers': '40',
#             'checkers': '20',
#             'dir_cache_time': '90m',
#             'attr_timeout': '45s',
#             'low_level_retries': '12',
#             'retries': '4',
#             'contimeout': '45s',
#             'timeout': '240s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 10,
#             'prefetch_factor': 5,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     }, # Adding from here

#     {
#         'run_id': 'rclone_full_workers8_swift',
#         'description': 'Full rclone cache, 8 workers',
#         'rclone_remote': 'chi_uc',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '20G',
#             'vfs_cache_max_age': '5h',
#             'vfs_read_chunk_size': '256M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '2G',
#             'buffer_size': '512M',
#             'transfers': '32',
#             'checkers': '16',
#             'dir_cache_time': '1h',
#             'attr_timeout': '30s',
#             'low_level_retries': '10',
#             'retries': '3',
#             'contimeout': '30s',
#             'timeout': '120s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 8,
#             'prefetch_factor': 4,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_aggressive_workers12_swift',
#         'description': 'Aggressive rclone settings, 12 workers',
#         'rclone_remote': 'chi_uc',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '50G',
#             'vfs_cache_max_age': '8h',
#             'vfs_read_chunk_size': '512M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '4G',
#             'buffer_size': '1G',
#             'transfers': '48',
#             'checkers': '24',
#             'dir_cache_time': '2h',
#             'attr_timeout': '60s',
#             'low_level_retries': '15',
#             'retries': '5',
#             'contimeout': '60s',
#             'timeout': '300s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 12,
#             'prefetch_factor': 6,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_balanced_optimized_swift',
#         'description': 'Balanced optimized configuration',
#         'rclone_remote': 'chi_uc',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '35G',
#             'vfs_cache_max_age': '6h',
#             'vfs_read_chunk_size': '384M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '3G',
#             'buffer_size': '768M',
#             'transfers': '40',
#             'checkers': '20',
#             'dir_cache_time': '90m',
#             'attr_timeout': '45s',
#             'low_level_retries': '12',
#             'retries': '4',
#             'contimeout': '45s',
#             'timeout': '240s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 10,
#             'prefetch_factor': 5,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     },
#     {
#         'run_id': 'rclone_balanced_streaming_swift',
#         'description': 'Balanced optimized with streaming and no-modtime',
#         'rclone_remote': 'chi_uc',
#         'rclone_container': 'ansh-lab4-tests-bucket',
#         'rclone_options': {
#             'vfs_cache_mode': 'full',
#             'vfs_cache_max_size': '35G',
#             'vfs_cache_max_age': '6h',
#             'vfs_read_chunk_size': '384M',
#             'vfs_read_chunk_size_limit': 'off',
#             'vfs_read_ahead': '3G',
#             'vfs_read_chunk_streams': '32',
#             'no_modtime': True,
#             'buffer_size': '768M',
#             'transfers': '40',
#             'checkers': '20',
#             'dir_cache_time': '90m',
#             'attr_timeout': '45s',
#             'low_level_retries': '12',
#             'retries': '4',
#             'contimeout': '45s',
#             'timeout': '240s',
#         },
#         'dataloader_options': {
#             'batch_size': 64,
#             'num_workers': 10,
#             'prefetch_factor': 5,
#             'pin_memory': True,
#         },
#         'epochs': 5
#     }
    
# ]

BENCHMARK_CONFIGURATIONS = [
    # {
    #     'run_id': 'rclone_aggressive_workers12',
    #     'description': 'Aggressive rclone settings, 12 workers',
    #     'rclone_remote': 'rclone_s3',
    #     'rclone_container': 'ansh-lab4-tests-bucket',
    #     'rclone_options': {
    #         'vfs_cache_mode': 'full',
    #         'vfs_cache_max_size': '50G',
    #         'vfs_cache_max_age': '8h',
    #         'vfs_read_chunk_size': '512M',
    #         'vfs_read_chunk_size_limit': 'off',
    #         'vfs_read_ahead': '4G',
    #         'buffer_size': '1G',
    #         'transfers': '48',
    #         'checkers': '24',
    #         'dir_cache_time': '2h',
    #         'attr_timeout': '60s',
    #         'low_level_retries': '15',
    #         'retries': '5',
    #         'contimeout': '60s',
    #         'timeout': '300s',
    #     },
    #     'dataloader_options': {
    #         'batch_size': 64,
    #         'num_workers': 8,
    #         'prefetch_factor': 4,
    #         'pin_memory': True,
    #     },
    #     'epochs': 5
    # }

    # {
    #     'run_id': 'rclone_aggressive_workers12_1',
    #     'description': 'Aggressive rclone settings, 12 workers',
    #     'rclone_remote': 'chi_uc',
    #     'rclone_container': 'ansh-lab4-tests-bucket',
    #     'rclone_options': {
    #         'vfs_cache_mode': 'full',
    #         'vfs_cache_max_size': '50G',
    #         'vfs_cache_max_age': '8h',
    #         'vfs_read_chunk_size': '512M',
    #         'vfs_read_chunk_size_limit': 'off',
    #         'vfs_read_ahead': '4G',
    #         'buffer_size': '1G',
    #         'transfers': '48',
    #         'checkers': '24',
    #         'dir_cache_time': '2h',
    #         'attr_timeout': '60s',
    #         'low_level_retries': '15',
    #         'retries': '5',
    #         'contimeout': '60s',
    #         'timeout': '300s',
    #     },
    #     'dataloader_options': {
    #         'batch_size': 64,
    #         'num_workers': 32,
    #         'prefetch_factor': 128,
    #         'pin_memory': True,
    #     },
    #     'epochs': 5
    # }

    # {
    #     'run_id': 'rclone_aggressive_workers12_2',
    #     'description': 'Aggressive rclone settings, 12 workers',
    #     'rclone_remote': 'rclone_s3',
    #     'rclone_container': 'ansh-lab4-tests-bucket',
    #     'rclone_options': {
    #         'vfs_cache_mode': 'full',
    #         'vfs_cache_max_size': '50G',
    #         'vfs_cache_max_age': '8h',
    #         'vfs_read_chunk_size': '4M',
    #         'vfs_read_chunk_size_limit': 'off',
    #         'vfs_read_ahead': '128M',
    #         'buffer_size': '32M',
    #         'transfers': '256',
    #         'checkers': '48',
    #         'dir_cache_time': '2h',
    #         'attr_timeout': '60s',
    #         'low_level_retries': '15',
    #         'retries': '5',
    #         'contimeout': '60s',
    #         'timeout': '300s',
    #     },
    #     'dataloader_options': {
    #         'batch_size': 64,
    #         'num_workers': 48,
    #         'prefetch_factor': 256,
    #         'pin_memory': True,
    #     },
    #     'epochs': 5
    # }
    {
        'run_id': 'rclone_aggressive_workers12_1',
        'description': 'Aggressive rclone settings, 12 workers',
        'rclone_remote': 'rclone_s3',
        'rclone_container': 'ansh-lab4-tests-bucket',
        'rclone_options': {
            'vfs_cache_mode': 'full',
            'vfs_cache_max_size': '50G',
            'vfs_cache_max_age': '8h',
            'vfs_read_chunk_size': '4M',
            'vfs_read_chunk_size_limit': 'off',
            'vfs_read_ahead': '128M',
            'buffer_size': '32M',
            'transfers': '128',
            'checkers': '48',
            'dir_cache_time': '2h',
            'attr_timeout': '60s',
            'low_level_retries': '15',
            'retries': '5',
            'contimeout': '60s',
            'timeout': '300s',
        },
        'dataloader_options': {
            'batch_size': 64,
            'num_workers': 96,
            'prefetch_factor': 128,
            'pin_memory': True,
        },
        'epochs': 5
    }
    
]

def setup_logging(run_id):
    log_dir = "logs/benchmark_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{run_id}.log")
    
    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
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
    logger.info(f"Starting benchmark run: {run_id}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def cleanup_existing_processes(logger=None):
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
        'vfs_read_chunk_streams': '--vfs-read-chunk-streams',
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
    
    if options.get('no_modtime'):
        cmd += " --no-modtime"
    
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

def save_results_to_csv(results, rclone_options, logger=None):
    """Save results to CSV file"""
    results_file = RESULTS_CSV
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'run_id', 'description', 'total_training_time_sec', 'epochs', 
            'final_training_loss', 'best_val_accuracy', 'final_val_accuracy', 'best_val_loss',
            'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_inference_time_sec',
            'avg_epoch_time_sec', 'avg_data_loading_time_sec', 'avg_compute_time_sec', 
            'data_loading_ratio', 'samples_per_sec_training', 'total_samples_processed',
            'batch_size', 'num_workers', 'total_train_samples', 'total_val_samples', 'total_test_samples', 
            'trainable_parameters', 'vfs_cache_mode', 'vfs_cache_max_size', 'vfs_read_chunk_size', 
            'buffer_size', 'transfers', 'prefetch_factor', 'pin_memory'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Prepare row data
        row_data = {
            'timestamp': results['timestamp'],
            'run_id': results['run_id'],
            'description': results.get('description', ''),
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
            'vfs_cache_mode': rclone_options.get('vfs_cache_mode', 'default'),
            'vfs_cache_max_size': rclone_options.get('vfs_cache_max_size', 'default'),
            'vfs_read_chunk_size': rclone_options.get('vfs_read_chunk_size', 'default'),
            'buffer_size': rclone_options.get('buffer_size', 'default'),
            'transfers': rclone_options.get('transfers', 'default'),
            'prefetch_factor': results.get('prefetch_factor', 'default'),
            'pin_memory': results.get('pin_memory', 'default'),
        }
        
        writer.writerow(row_data)
    
    if logger:
        logger.info(f"Results appended to {results_file}")

def run_single_benchmark(config):
    """Run a single benchmark configuration"""
    run_id = config['run_id']
    description = config['description']
    rclone_remote = config['rclone_remote']
    rclone_container = config['rclone_container']
    rclone_options = config['rclone_options']
    dataloader_options = config['dataloader_options']
    epochs = config['epochs']
    
    # Setup logging for this specific run
    logger = setup_logging(run_id)
    
    try:
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("="*80)
        logger.info(f"BENCHMARK CONFIGURATION: {run_id}")
        logger.info(f"Description: {description}")
        logger.info("="*80)
        
        logger.info("Configuration:")
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Remote: {rclone_remote}:{rclone_container}")
        logger.info(f"  Cache Mode: {rclone_options.get('vfs_cache_mode', 'default')}")
        logger.info(f"  Cache Size: {rclone_options.get('vfs_cache_max_size', 'default')}")
        logger.info(f"  Batch Size: {dataloader_options.get('batch_size', 64)}")
        logger.info(f"  Num Workers: {dataloader_options.get('num_workers', 8)}")
        logger.info(f"  Prefetch Factor: {dataloader_options.get('prefetch_factor', 2)}")
        logger.info(f"  Pin Memory: {dataloader_options.get('pin_memory', True)}")
        logger.info(f"  Epochs: {epochs}")
        
        # Build mount command
        mount_cmd = build_mount_command(rclone_remote, rclone_container, MOUNT_POINT, rclone_options)
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
            return None
        
        # Mount with new configuration
        logger.info(f"Mounting {rclone_remote}:{rclone_container} to {MOUNT_POINT}...")
        result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Mount failed!")
            logger.error(f"stderr: {result.stderr}")
            return None
        else:
            # Wait for mount to be ready
            time.sleep(5)
            
            # Verify mount
            if os.path.exists(MOUNT_POINT) and os.listdir(MOUNT_POINT):
                logger.info(f"Mounted successfully!")
                logger.info(f"Contents: {os.listdir(MOUNT_POINT)}")
            else:
                logger.warning("Mount point exists but appears empty")
                return None
        
        # Check data directory
        DATA_DIR = MOUNT_POINT
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info(f"Contents: {os.listdir(DATA_DIR)}")
        
        # Run training benchmark
        logger.info("="*60)
        logger.info(f"TRAINING BENCHMARK RUN: {run_id}")
        logger.info("="*60)
        
        # Record timestamp before training starts
        start_timestamp = datetime.now().isoformat()
        
        results = run_training_benchmark(
            data_dir=DATA_DIR,
            batch_size=dataloader_options.get('batch_size', 64),
            num_workers=dataloader_options.get('num_workers', 8),
            epochs=epochs,
            prefetch_factor=dataloader_options.get('prefetch_factor', 2),
            pin_memory=dataloader_options.get('pin_memory', True),
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
        
        # Add metadata to results
        results['timestamp'] = start_timestamp
        results['run_id'] = run_id
        results['description'] = description
        results['rclone_config'] = rclone_options
        results['prefetch_factor'] = dataloader_options.get('prefetch_factor', 2)
        results['pin_memory'] = dataloader_options.get('pin_memory', True)
        
        save_results_to_csv(results, rclone_options, logger)
        logger.info(f"Results saved with Run ID: {run_id}")
        
        logger.info("="*60)
        logger.info(f"Benchmark {run_id} completed successfully!")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark {run_id} failed with error: {e}")
        return None
    
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
        
        # Wait between configurations
        time.sleep(5)

def main():
    """Main function to run multiple benchmark configurations"""
    
    print("="*80)
    print("MULTIPLE BENCHMARK CONFIGURATIONS")
    print("="*80)
    print(f"Total configurations to run: {len(BENCHMARK_CONFIGURATIONS)}")
    
    results_summary = []
    
    for i, config in enumerate(BENCHMARK_CONFIGURATIONS, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING CONFIGURATION {i}/{len(BENCHMARK_CONFIGURATIONS)}: {config['run_id']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")
        
        result = run_single_benchmark(config)
        
        if result:
            results_summary.append({
                'run_id': config['run_id'],
                'description': config['description'],
                'success': True,
                'throughput': result['samples_per_sec_training'],
                'accuracy': result['test_accuracy'],
                'data_loading_ratio': result['data_loading_ratio']
            })
            print(f"✅ Configuration {config['run_id']} completed successfully")
        else:
            results_summary.append({
                'run_id': config['run_id'],
                'description': config['description'],
                'success': False,
                'throughput': 0,
                'accuracy': 0,
                'data_loading_ratio': 0
            })
            print(f"❌ Configuration {config['run_id']} failed")
        
        # Wait between configurations
        if i < len(BENCHMARK_CONFIGURATIONS):
            print(f"Waiting 10 seconds before next configuration...")
            time.sleep(10)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY")
    print("="*80)
    
    successful_runs = [r for r in results_summary if r['success']]
    
    if successful_runs:
        print(f"Successful runs: {len(successful_runs)}/{len(BENCHMARK_CONFIGURATIONS)}")
        print("\nRanking by throughput (samples/sec):")
        sorted_by_throughput = sorted(successful_runs, key=lambda x: x['throughput'], reverse=True)
        
        for i, result in enumerate(sorted_by_throughput, 1):
            print(f"{i:2d}. {result['run_id']:25s} | {result['throughput']:8.1f} samples/sec | "
                  f"Accuracy: {result['accuracy']:5.1f}% | Data ratio: {result['data_loading_ratio']:5.1%}")
        
        print(f"\nBest throughput: {sorted_by_throughput[0]['run_id']} with {sorted_by_throughput[0]['throughput']:.1f} samples/sec")
        
        # Find best accuracy
        sorted_by_accuracy = sorted(successful_runs, key=lambda x: x['accuracy'], reverse=True)
        print(f"Best accuracy: {sorted_by_accuracy[0]['run_id']} with {sorted_by_accuracy[0]['accuracy']:.1f}%")
        
    else:
        print("❌ No successful runs!")
    
    print("="*80)
    print("All benchmarks completed!")
    print(f"Results saved to: {RESULTS_CSV}")
    print("="*80)

if __name__ == "__main__":
    main()


