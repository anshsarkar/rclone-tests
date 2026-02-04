#!/usr/bin/env python3

# python gpu_monitor.py --output gpu_utilization.csv --run-id test_001
import argparse
import csv
import subprocess
import time
import sys
from datetime import datetime
import psutil


def get_system_stats():
    """Query nvidia-smi for GPU stats and get CPU utilization."""
    # Get GPU stats
    gpu_stats = None
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
        
        if result.returncode == 0:
            line = result.stdout.strip()
            if line:
                parts = [p.strip() for p in line.split(',')]
                
                gpu_stats = {
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
    except FileNotFoundError:
        print("Warning: nvidia-smi not found. GPU monitoring disabled.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: GPU monitoring error: {e}", file=sys.stderr)
    
    # Get CPU stats
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Short interval for current reading
        memory = psutil.virtual_memory()
        cpu_stats = {
            'cpu_util_percent': cpu_percent,
            'memory_util_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
        }
    except Exception as e:
        print(f"Warning: CPU monitoring error: {e}", file=sys.stderr)
        cpu_stats = {
            'cpu_util_percent': None,
            'memory_util_percent': None,
            'memory_used_gb': None,
            'memory_total_gb': None,
        }
    
    # Get Network stats
    try:
        net_io = psutil.net_io_counters()
        network_stats = {
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'network_packets_sent': net_io.packets_sent,
            'network_packets_recv': net_io.packets_recv,
        }
    except Exception as e:
        print(f"Warning: Network monitoring error: {e}", file=sys.stderr)
        network_stats = {
            'network_bytes_sent': None,
            'network_bytes_recv': None,
            'network_packets_sent': None,
            'network_packets_recv': None,
        }
    
    return gpu_stats, cpu_stats, network_stats


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU and CPU and CPU utilization')
    parser.add_argument('--output', '-o', default='gpu_utilization.csv',
                        help='Output CSV file (default: gpu_utilization.csv)')
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                        help='Sampling interval in seconds (default: 0.5)')
    parser.add_argument('--run-id', '-r', default='',
                        help='Run ID to tag samples with (optional)')
    args = parser.parse_args()
    
    # Check if system monitoring is available
    gpu_stats, cpu_stats, network_stats = get_system_stats()
    if gpu_stats is None and cpu_stats['cpu_util_percent'] is None:
        print("No GPU or CPU monitoring available. Exiting.")
        sys.exit(1)
    
    has_gpu = gpu_stats is not None
    has_cpu = cpu_stats['cpu_util_percent'] is not None
    has_network = network_stats['network_bytes_sent'] is not None
    
    print(f"System Monitor Started")
    print(f"======================")
    print(f"Output file: {args.output}")
    print(f"Interval: {args.interval}s")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    print(f"GPU monitoring: {'✓' if has_gpu else '✗'}")
    print(f"CPU monitoring: {'✓' if has_cpu else '✗'}")
    print(f"Network monitoring: {'✓' if has_network else '✗'}")
    print(f"\nPress Ctrl+C to stop monitoring.\n")
    
    # CSV headers - include GPU, CPU, and Network metrics
    headers = [
        'timestamp', 'run_id', 'gpu_util_percent', 'mem_util_percent',
        'mem_used_mb', 'mem_total_mb', 'temperature_c', 'power_w',
        'cpu_util_percent', 'system_memory_util_percent', 
        'system_memory_used_gb', 'system_memory_total_gb',
        'network_bytes_sent', 'network_bytes_recv', 'network_packets_sent', 'network_packets_recv',
        'network_mbps_sent', 'network_mbps_recv'
    ]
    
    # Check if file exists to determine if we need headers
    file_exists = False
    try:
        with open(args.output, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    sample_count = 0
    
    # Lists to store values for average calculation
    gpu_utils = []
    mem_utils = []
    cpu_utils = []
    system_mem_utils = []
    temperatures = []
    power_draws = []
    network_sent_rates = []
    network_recv_rates = []
    
    # Track previous network stats for rate calculation
    prev_network_stats = None
    prev_time = time.time()
    
    try:
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists:
                writer.writerow(headers)
                f.flush()
            
            while True:
                current_time = time.time()
                gpu_stats, cpu_stats, network_stats = get_system_stats()
                
                # Calculate network rates (bytes per second)
                network_mbps_sent = None
                network_mbps_recv = None
                if has_network and prev_network_stats and network_stats['network_bytes_sent'] is not None:
                    time_diff = current_time - prev_time
                    if time_diff > 0:
                        bytes_sent_diff = network_stats['network_bytes_sent'] - prev_network_stats['network_bytes_sent']
                        bytes_recv_diff = network_stats['network_bytes_recv'] - prev_network_stats['network_bytes_recv']
                        
                        # Convert to Mbps (Megabits per second)
                        network_mbps_sent = (bytes_sent_diff * 8) / (time_diff * 1_000_000)
                        network_mbps_recv = (bytes_recv_diff * 8) / (time_diff * 1_000_000)
                
                # Collect data for CSV
                row = [
                    datetime.now().isoformat(),
                    args.run_id,
                    gpu_stats['gpu_util_percent'] if gpu_stats else None,
                    gpu_stats['mem_util_percent'] if gpu_stats else None,
                    gpu_stats['mem_used_mb'] if gpu_stats else None,
                    gpu_stats['mem_total_mb'] if gpu_stats else None,
                    gpu_stats['temperature_c'] if gpu_stats else None,
                    gpu_stats['power_w'] if gpu_stats else None,
                    cpu_stats['cpu_util_percent'],
                    cpu_stats['memory_util_percent'],
                    cpu_stats['memory_used_gb'],
                    cpu_stats['memory_total_gb'],
                    network_stats['network_bytes_sent'],
                    network_stats['network_bytes_recv'],
                    network_stats['network_packets_sent'],
                    network_stats['network_packets_recv'],
                    network_mbps_sent,
                    network_mbps_recv,
                ]
                writer.writerow(row)
                f.flush()
                
                sample_count += 1
                
                # Collect values for averages (only if not None)
                if gpu_stats:
                    if gpu_stats['gpu_util_percent'] is not None:
                        gpu_utils.append(gpu_stats['gpu_util_percent'])
                    if gpu_stats['mem_util_percent'] is not None:
                        mem_utils.append(gpu_stats['mem_util_percent'])
                    if gpu_stats['temperature_c'] is not None:
                        temperatures.append(gpu_stats['temperature_c'])
                    if gpu_stats['power_w'] is not None:
                        power_draws.append(gpu_stats['power_w'])
                
                if cpu_stats['cpu_util_percent'] is not None:
                    cpu_utils.append(cpu_stats['cpu_util_percent'])
                if cpu_stats['memory_util_percent'] is not None:
                    system_mem_utils.append(cpu_stats['memory_util_percent'])
                
                # Collect network rates for averages
                if network_mbps_sent is not None:
                    network_sent_rates.append(network_mbps_sent)
                if network_mbps_recv is not None:
                    network_recv_rates.append(network_mbps_recv)
                
                # Update previous network stats for next calculation
                prev_network_stats = network_stats
                prev_time = current_time
                
                # Print status every 10 samples
                if sample_count % 10 == 0:
                    gpu_util = gpu_stats['gpu_util_percent'] if gpu_stats else 0
                    mem_util = gpu_stats['mem_util_percent'] if gpu_stats else 0
                    temp = gpu_stats['temperature_c'] if gpu_stats else 0
                    power = gpu_stats['power_w'] if gpu_stats else 0
                    cpu_util = cpu_stats['cpu_util_percent'] or 0
                    sys_mem_util = cpu_stats['memory_util_percent'] or 0
                    
                    status_parts = [f"[{sample_count}]"]
                    if has_gpu:
                        status_parts.append(f"GPU: {gpu_util:5.1f}%")
                        status_parts.append(f"VRAM: {mem_util:5.1f}%")
                        status_parts.append(f"Temp: {temp:.0f}°C")
                        status_parts.append(f"Power: {power:.1f}W")
                    if has_cpu:
                        status_parts.append(f"CPU: {cpu_util:5.1f}%")
                        status_parts.append(f"RAM: {sys_mem_util:5.1f}%")
                    if has_network and network_mbps_sent is not None:
                        status_parts.append(f"Net: ↑{network_mbps_sent:.1f} ↓{network_mbps_recv:.1f} Mbps")
                    
                    print(" | ".join(status_parts))
                
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")
        print(f"Total samples: {sample_count}")
        
        # Calculate and print averages
        print(f"\n{'='*60}")
        print("AVERAGE UTILIZATION SUMMARY")
        print(f"{'='*60}")
        
        if gpu_utils:
            print(f"GPU Utilization: {sum(gpu_utils)/len(gpu_utils):6.2f}% (avg over {len(gpu_utils)} samples)")
        if mem_utils:
            print(f"GPU Memory:      {sum(mem_utils)/len(mem_utils):6.2f}% (avg over {len(mem_utils)} samples)")
        if temperatures:
            print(f"GPU Temperature: {sum(temperatures)/len(temperatures):6.1f}°C (avg over {len(temperatures)} samples)")
        if power_draws:
            print(f"GPU Power:       {sum(power_draws)/len(power_draws):6.1f}W (avg over {len(power_draws)} samples)")
        
        if cpu_utils:
            print(f"CPU Utilization: {sum(cpu_utils)/len(cpu_utils):6.2f}% (avg over {len(cpu_utils)} samples)")
        if system_mem_utils:
            print(f"System Memory:   {sum(system_mem_utils)/len(system_mem_utils):6.2f}% (avg over {len(system_mem_utils)} samples)")
        
        if network_sent_rates:
            print(f"Network Upload:  {sum(network_sent_rates)/len(network_sent_rates):6.2f} Mbps (avg over {len(network_sent_rates)} samples)")
        if network_recv_rates:
            print(f"Network Download:{sum(network_recv_rates)/len(network_recv_rates):6.2f} Mbps (avg over {len(network_recv_rates)} samples)")
        
        print(f"{'='*60}")
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
