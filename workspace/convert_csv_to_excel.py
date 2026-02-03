# Convert csv file to excel file

import pandas as pd
def convert_csv_to_excel(csv_file, excel_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # filter out columns that are not needed, we need run_id, training_time_sec, epochs, samples per sec train, batch size, number of workers, all rclone options and average gpu utilization
    columns_to_keep = [
        'run_id', 'total_training_time_sec', 'epochs', 'samples_per_sec_training',
        'batch_size', 'num_workers', 'vfs_cache_mode', 'vfs_cache_max_size', 
        'vfs_read_chunk_size', 'buffer_size', 'transfers', 'avg_gpu_utilization'
    ]
    
    # Filter the dataframe to keep only specified columns that exist
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df_filtered = df[available_columns]
    
    # Write to an Excel file
    df_filtered.to_excel(excel_file, index=False)

if __name__ == "__main__":
    csv_file = 'final_benchmark_results.csv'
    excel_file = 'final_benchmark_results.xlsx'
    convert_csv_to_excel(csv_file, excel_file)
    print(f"Converted {csv_file} to {excel_file}")