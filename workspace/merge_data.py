#!/usr/bin/env python3

import pandas as pd
import os

# File paths
RESULTS_CSV = 'multiple_benchmark_results.csv'
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

def calculate_gpu_metrics_by_timestamp(df_benchmark, df_gpu_clean):
    if df_benchmark.empty or df_gpu_clean.empty:
        print("No benchmark data or GPU data available for timestamp-based metrics calculation")
        return pd.DataFrame()
    
    print(f"\nCalculating GPU metrics using timestamp-based attribution...")
    
    # Convert timestamps to datetime
    df_benchmark = df_benchmark.copy()
    df_gpu_clean = df_gpu_clean.copy()
    
    df_benchmark['timestamp'] = pd.to_datetime(df_benchmark['timestamp'])
    df_gpu_clean['timestamp'] = pd.to_datetime(df_gpu_clean['timestamp'])
    
    # Sort benchmark data by timestamp to get proper time ranges
    df_benchmark = df_benchmark.sort_values('timestamp')
    df_gpu_clean = df_gpu_clean.sort_values('timestamp')
    
    print(f"Benchmark runs timestamp range: {df_benchmark['timestamp'].min()} to {df_benchmark['timestamp'].max()}")
    print(f"GPU data timestamp range: {df_gpu_clean['timestamp'].min()} to {df_gpu_clean['timestamp'].max()}")
    
    # Create list to store GPU metrics for each run
    gpu_metrics_list = []
    
    for i, row in df_benchmark.iterrows():
        run_id = row['run_id']
        start_time = row['timestamp']
        
        # Find the end time for this run (start of next run or end of GPU data)
        if i < len(df_benchmark) - 1:
            # Use start of next run as end time
            end_time = df_benchmark.iloc[i + 1]['timestamp']
        else:
            # For last run, use end of GPU data
            end_time = df_gpu_clean['timestamp'].max()
        
        print(f"Run {run_id}: {start_time} to {end_time}")
        
        # Filter GPU data for this time range
        gpu_run_data = df_gpu_clean[
            (df_gpu_clean['timestamp'] >= start_time) & 
            (df_gpu_clean['timestamp'] < end_time)
        ]
        
        if len(gpu_run_data) == 0:
            print(f"  WARNING: No GPU data found for run {run_id}")
            # Add empty metrics for this run
            gpu_metrics_list.append({
                'run_id': run_id,
                'avg_gpu_utilization': None,
                'max_gpu_utilization': None,
                'min_gpu_utilization': None,
                'gpu_utilization_std': None,
                'avg_memory_used_mb': None,
                'max_memory_used_mb': None,
                'total_memory_mb': None,
                'avg_memory_usage_percent': None,
                'max_memory_usage_percent': None,
                'gpu_sample_count': 0
            })
            continue
        
        print(f"  Found {len(gpu_run_data)} GPU samples for run {run_id}")
        
        # Calculate metrics for this run
        metrics = {
            'run_id': run_id,
            'avg_gpu_utilization': gpu_run_data['gpu_util_percent'].mean(),
            'max_gpu_utilization': gpu_run_data['gpu_util_percent'].max(),
            'min_gpu_utilization': gpu_run_data['gpu_util_percent'].min(),
            'gpu_utilization_std': gpu_run_data['gpu_util_percent'].std(),
            'avg_memory_used_mb': gpu_run_data['mem_used_mb'].mean(),
            'max_memory_used_mb': gpu_run_data['mem_used_mb'].max(),
            'total_memory_mb': gpu_run_data['mem_total_mb'].iloc[0] if len(gpu_run_data) > 0 else None,
            'gpu_sample_count': len(gpu_run_data)
        }
        
        # Calculate memory usage percentages
        if metrics['total_memory_mb'] is not None and metrics['total_memory_mb'] > 0:
            metrics['avg_memory_usage_percent'] = (metrics['avg_memory_used_mb'] / metrics['total_memory_mb'] * 100)
            metrics['max_memory_usage_percent'] = (metrics['max_memory_used_mb'] / metrics['total_memory_mb'] * 100)
        else:
            metrics['avg_memory_usage_percent'] = None
            metrics['max_memory_usage_percent'] = None
        
        # Add temperature if available
        if 'temperature_c' in gpu_run_data.columns:
            metrics['avg_temperature_c'] = gpu_run_data['temperature_c'].mean()
            metrics['max_temperature_c'] = gpu_run_data['temperature_c'].max()
        
        gpu_metrics_list.append(metrics)
    
    # Convert to DataFrame
    gpu_metrics = pd.DataFrame(gpu_metrics_list)
    
    # Round numerical columns
    numeric_cols = gpu_metrics.select_dtypes(include=['float64', 'int64']).columns
    gpu_metrics[numeric_cols] = gpu_metrics[numeric_cols].round(2)
    
    print(f"Calculated GPU metrics for {len(gpu_metrics)} runs using timestamp attribution")
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
        
        # Calculate GPU metrics using timestamp-based attribution
        gpu_metrics = calculate_gpu_metrics_by_timestamp(df_benchmark, df_gpu_clean)
        
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