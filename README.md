# Rclone Tests

Performance benchmarking and testing suite for different rclone configurations and data loading strategies.

## Quick Start

### Benchmarking
**Important**: Before running benchmarks, start GPU monitoring in a separate terminal:
```bash
python workspace/gpu_monitor.py --output gpu_utilization.csv --run-id <your_run_id>
```

Then run benchmarks:
```bash
python workspace/run_multiple_benchmarks.py
```

## Workspace Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Basic PyTorch training with local data |
| `train_2.py` | Enhanced training script with better monitoring |
| `train_s3.py` | Training with rclone-mounted S3 storage |
| `train_s3_lit.py` | PyTorch Lightning training with LitData streaming from S3 |
| `run_benchmark.py` | Single benchmark execution framework |
| `run_multiple_benchmarks.py` | Automated multiple benchmark configurations |
| `gpu_monitor.py` | GPU utilization monitoring during training |
| `prepare_litdata_food11.py` | Data sharding preparation for LitData |
| `convert_csv_to_excel.py` | Convert benchmark results to Excel format |
| `merge_data.py` | Utility for merging multiple result files |

## Important Notes

- **Data Sharding**: Before using `train_s3_lit.py`, ensure data is properly sharded and stored using `prepare_litdata_food11.py`
- **GPU Monitoring**: Always run `gpu_monitor.py` in a separate shell before executing `run_multiple_benchmarks.py` for proper performance tracking
- **Initial Results**: Preliminary benchmark results are available in `final_benchmark_results.csv`
- **Detailed Logs**: For result verification and debugging, check the `workspace/logs/` directory which contains:
  - `benchmark_logs/`: Individual run logs for each benchmark configuration
  - `torch_s3/`: Training logs and metrics from S3-based experiments
- **Helper Scripts**: Additional utility scripts are available in the workspace directory for data processing and result analysis

## Jupyter Notebooks

The repository includes several Jupyter notebooks for interactive setup and configuration of different storage backends and compute environments.