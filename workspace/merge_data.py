#!/usr/bin/env python3

import pandas as pd
import os

# File paths
RESULTS_CSV = 'training_benchmark_results.csv'
GPU_CSV = 'gpu_utilization.csv'
FINAL_CSV = 'final_benchmark_results.csv'

def load_data():
    print("=" * 60)
    print("DATA PROCESSING AND MERGE")
    print("=" * 60)
    
    # Read benchmark results
    if os.path.exists(RESULTS_CSV):
        df_benchmark = pd.read_csv(RESULTS_CSV)
        print(f"Loaded benchmark data: {len(df_benchmark)} runs")
        print(f"Columns: {list(df_benchmark.columns)}")
    else:
        print(f"ERROR: {RESULTS_CSV} not found!")
        df_benchmark = pd.DataFrame()
    
    # Read GPU utilization data  
    if os.path.exists(GPU_CSV):
        df_gpu = pd.read_csv(GPU_CSV)
        print(f"Loaded GPU data: {len(df_gpu)} samples")
        print(f"Columns: {list(df_gpu.columns)}")
    else:
        print(f"WARNING: {GPU_CSV} not found - will proceed with benchmark data only")
        df_gpu = pd.DataFrame()
    
    return df_benchmark, df_gpu

def clean_gpu_data(df_gpu):
    if df_gpu.empty:
        print("No GPU data to clean")
        return pd.DataFrame()
    
    print(f"\nCleaning GPU data...")
    print(f"Original GPU samples: {len(df_gpu)}")
    
    # Remove entries with 0% GPU utilization
    if 'gpu_util_percent' in df_gpu.columns:
        df_gpu_clean = df_gpu[df_gpu['gpu_util_percent'] > 0].copy()
        print(f"After removing 0% utilization: {len(df_gpu_clean)} samples")
        removed = len(df_gpu) - len(df_gpu_clean)
        print(f"Removed {removed} zero-utilization samples ({removed/len(df_gpu)*100:.1f}%)")
        
        if len(df_gpu_clean) == 0:
            print("WARNING: No valid GPU data after cleaning!")
            return pd.DataFrame()
        
        return df_gpu_clean
    else:
        print("WARNING: No 'gpu_util_percent' column found in GPU data")
        return df_gpu.copy()

def calculate_gpu_metrics(df_gpu_clean):
    if df_gpu_clean.empty or 'run_id' not in df_gpu_clean.columns:
        print("No GPU data available for metrics calculation")
        return pd.DataFrame()
    
    print(f"\nCalculating GPU metrics per run...")
    
    # Group by run_id and calculate aggregated metrics
    gpu_metrics = df_gpu_clean.groupby('run_id').agg({
        'gpu_util_percent': ['mean', 'max', 'std', 'min'],
        'mem_used_mb': ['mean', 'max'],
        'mem_total_mb': 'first',  # Should be constant
        'temperature_c': ['mean', 'max'] if 'temperature_c' in df_gpu_clean.columns else ['mean', 'max'],
        'timestamp': ['first', 'last', 'count']  # For tracking duration and sample count
    }).round(2)
    
    # Flatten column names
    gpu_metrics.columns = [f"gpu_{col[0]}_{col[1]}" if col[1] != 'first' else f"gpu_{col[0]}" 
                          for col in gpu_metrics.columns]
    
    # Rename columns for clarity
    gpu_metrics = gpu_metrics.rename(columns={
        'gpu_gpu_util_percent_mean': 'avg_gpu_utilization',
        'gpu_gpu_util_percent_max': 'max_gpu_utilization', 
        'gpu_gpu_util_percent_std': 'gpu_utilization_std',
        'gpu_gpu_util_percent_min': 'min_gpu_utilization',
        'gpu_mem_used_mb_mean': 'avg_memory_used_mb',
        'gpu_mem_used_mb_max': 'max_memory_used_mb',
        'gpu_mem_total_mb': 'total_memory_mb',
        'gpu_timestamp_count': 'gpu_sample_count'
    })
    
    # Calculate memory usage percentage
    gpu_metrics['avg_memory_usage_percent'] = (gpu_metrics['avg_memory_used_mb'] / gpu_metrics['total_memory_mb'] * 100).round(2)
    gpu_metrics['max_memory_usage_percent'] = (gpu_metrics['max_memory_used_mb'] / gpu_metrics['total_memory_mb'] * 100).round(2)
    
    # Reset index to make run_id a column
    gpu_metrics = gpu_metrics.reset_index()
    
    print(f"Calculated GPU metrics for {len(gpu_metrics)} runs")
    print(f"GPU metrics columns: {list(gpu_metrics.columns)}")
    
    return gpu_metrics

def merge_data(df_benchmark, gpu_metrics):
    if df_benchmark.empty:
        print("ERROR: No benchmark data available!")
        return pd.DataFrame()
    
    if not gpu_metrics.empty:
        print(f"\nMerging benchmark data with GPU metrics...")
        df_final = df_benchmark.merge(gpu_metrics, on='run_id', how='left')
        print(f"Merged data: {len(df_final)} runs")
        
        # Fill missing GPU values with NaN (for runs without GPU data)
        gpu_columns = [col for col in gpu_metrics.columns if col != 'run_id']
        missing_gpu_runs = df_final[gpu_columns[0]].isna().sum()
        if missing_gpu_runs > 0:
            print(f"Warning: {missing_gpu_runs} runs have no GPU data")
    else:
        print("No GPU metrics to merge - using benchmark data only")
        df_final = df_benchmark.copy()
    
    print(f"Final dataset columns: {len(df_final.columns)}")
    print(f"Final dataset shape: {df_final.shape}")
    
    return df_final

def save_final_dataset(df_final):
    if df_final.empty:
        print("ERROR: No data to save!")
        return False
    
    print(f"\nSaving final dataset to {FINAL_CSV}...")
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"SUCCESS: Saved {len(df_final)} runs to {FINAL_CSV}")
    
    # Display summary of the final dataset
    print(f"\nFinal dataset summary:")
    print(f"  Rows: {len(df_final)}")
    print(f"  Columns: {len(df_final.columns)}")
    
    # Show first few rows
    print(f"\nFirst 3 rows of final dataset:")
    display_cols = ['run_id', 'timestamp', 'total_training_time_sec', 'test_accuracy']
    if 'avg_gpu_utilization' in df_final.columns:
        display_cols.extend(['avg_gpu_utilization', 'max_gpu_utilization'])
    if 'avg_memory_usage_percent' in df_final.columns:
        display_cols.append('avg_memory_usage_percent')
    
    available_cols = [col for col in display_cols if col in df_final.columns]
    print(df_final[available_cols].head(3).to_string(index=False))
    
    print(f"\nData processing completed successfully!")
    print(f"Final file: {FINAL_CSV}")
    
    return True

def display_column_reference(df_final):
    if df_final.empty:
        return
    
    print("=" * 60)
    print("FINAL DATASET COLUMN REFERENCE")
    print("=" * 60)
    
    print("\nBenchmark columns:")
    benchmark_cols = [col for col in df_final.columns if not col.startswith(('avg_gpu', 'max_gpu', 'min_gpu', 'gpu_', 'total_memory', 'avg_memory', 'max_memory'))]
    for col in benchmark_cols:
        print(f"  {col}")
    
    if 'avg_gpu_utilization' in df_final.columns:
        print("\nGPU metrics columns:")
        gpu_cols = [col for col in df_final.columns if col.startswith(('avg_gpu', 'max_gpu', 'min_gpu', 'gpu_', 'total_memory', 'avg_memory', 'max_memory'))]
        for col in gpu_cols:
            print(f"  {col}")
    
    print(f"\nTotal columns: {len(df_final.columns)}")
    print("=" * 60)

def main():
    try:
        # Load data
        df_benchmark, df_gpu = load_data()
        
        # Clean GPU data
        df_gpu_clean = clean_gpu_data(df_gpu)
        
        # Calculate GPU metrics
        gpu_metrics = calculate_gpu_metrics(df_gpu_clean)
        
        # Merge data
        df_final = merge_data(df_benchmark, gpu_metrics)
        
        # Save final dataset
        success = save_final_dataset(df_final)
        
        if success:
            # Display column reference
            display_column_reference(df_final)
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"ERROR: Data processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())