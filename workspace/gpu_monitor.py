#!/usr/bin/env python3

# python gpu_monitor.py --output gpu_utilization.csv --run-id test_001
import argparse
import csv
import subprocess
import time
import sys
from datetime import datetime


def get_gpu_stats():
    """Query nvidia-smi for GPU stats."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=timestamp,gpu_uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
            
        line = result.stdout.strip()
        if not line:
            return None
            
        parts = [p.strip() for p in line.split(',')]
        
        return {
            'nvidia_timestamp': parts[0],
            'gpu_uuid': parts[1],
            'gpu_util_percent': float(parts[2]) if parts[2] != '[N/A]' else None,
            'mem_util_percent': float(parts[3]) if parts[3] != '[N/A]' else None,
            'mem_used_mb': float(parts[4]) if parts[4] != '[N/A]' else None,
            'mem_total_mb': float(parts[5]) if parts[5] != '[N/A]' else None,
            'temperature_c': float(parts[6]) if parts[6] != '[N/A]' else None,
            'power_w': float(parts[7]) if parts[7] != '[N/A]' else None,
        }
        
    except subprocess.TimeoutExpired:
        print("Warning: nvidia-smi timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Is NVIDIA driver installed?", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error querying GPU: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU utilization')
    parser.add_argument('--output', '-o', default='gpu_utilization.csv',
                        help='Output CSV file (default: gpu_utilization.csv)')
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                        help='Sampling interval in seconds (default: 0.5)')
    parser.add_argument('--run-id', '-r', default='',
                        help='Run ID to tag samples with (optional)')
    args = parser.parse_args()
    
    # Check if GPU is available
    stats = get_gpu_stats()
    if stats is None:
        print("No GPU available or nvidia-smi failed. Exiting.")
        sys.exit(1)
    
    print(f"GPU Monitor Started")
    print(f"==================")
    print(f"Output file: {args.output}")
    print(f"Interval: {args.interval}s")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    print(f"\nPress Ctrl+C to stop monitoring.\n")
    
    # CSV headers
    headers = [
        'timestamp', 'run_id', 'gpu_util_percent', 'mem_util_percent',
        'mem_used_mb', 'mem_total_mb', 'temperature_c', 'power_w'
    ]
    
    # Check if file exists to determine if we need headers
    file_exists = False
    try:
        with open(args.output, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    sample_count = 0
    
    try:
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists:
                writer.writerow(headers)
                f.flush()
            
            while True:
                stats = get_gpu_stats()
                
                if stats:
                    row = [
                        datetime.now().isoformat(),
                        args.run_id,
                        stats['gpu_util_percent'],
                        stats['mem_util_percent'],
                        stats['mem_used_mb'],
                        stats['mem_total_mb'],
                        stats['temperature_c'],
                        stats['power_w'],
                    ]
                    writer.writerow(row)
                    f.flush()
                    
                    sample_count += 1
                    
                    # Print status every 10 samples
                    if sample_count % 10 == 0:
                        gpu_util = stats['gpu_util_percent'] or 0
                        mem_util = stats['mem_util_percent'] or 0
                        temp = stats['temperature_c'] or 0
                        power = stats['power_w'] or 0
                        print(f"[{sample_count}] GPU: {gpu_util:5.1f}% | Mem: {mem_util:5.1f}% | Temp: {temp:.0f}°C | Power: {power:.1f}W")
                
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")
        print(f"Total samples: {sample_count}")
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
