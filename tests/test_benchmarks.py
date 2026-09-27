"""
Performance Benchmarks:
1. Per-sample and batch inference latency (ms)
2. Memory allocation and peak memory usage (MB)
3. Processing throughput (samples / second)
"""

import unittest
import time
import tracemalloc
import numpy as np
import pandas as pd

from aws_weather_preprocessor import (
    AWSCorrectionAndAlertingPipeline,
    AtmosphericStateKalmanFilter,
    AWSDataPreprocessor,
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


class TestPerformanceBenchmarks(unittest.TestCase):

    def setUp(self):
        # 7-day dataset = 2016 samples
        config = SyntheticDataConfig(duration_days=7.0, frequency_minutes=5, inject_anomalies=False, random_seed=42)
        self.raw_df, _ = generate_synthetic_aws_data(config)

    def test_kalman_filter_latency_and_throughput(self):
        kf = AtmosphericStateKalmanFilter()
        
        tracemalloc.start()
        t0 = time.perf_counter()
        
        output = kf.filter_dataframe(self.raw_df)
        
        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time_sec = t1 - t0
        n_samples = len(self.raw_df)
        latency_per_sample_ms = (total_time_sec / n_samples) * 1000.0
        throughput_sps = n_samples / total_time_sec
        peak_mem_mb = peak_mem / (1024 * 1024)

        print(f"\n[Kalman Filter Benchmark] Samples: {n_samples:,} | Latency: {latency_per_sample_ms:.4f} ms/sample | Throughput: {throughput_sps:.1f} samples/s | Peak Mem: {peak_mem_mb:.2f} MB")

        # Assertions
        self.assertLess(latency_per_sample_ms, 2.0)  # Must be fast (< 2ms / sample)
        self.assertGreater(throughput_sps, 500)      # > 500 samples/sec
        self.assertLess(peak_mem_mb, 50.0)           # < 50 MB memory usage

    def test_end_to_end_pipeline_benchmark(self):
        pipeline = AWSCorrectionAndAlertingPipeline()

        tracemalloc.start()
        t0 = time.perf_counter()

        result = pipeline.process(self.raw_df, station_id="BENCHMARK-STATION-01")

        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time_sec = t1 - t0
        n_samples = len(self.raw_df)
        latency_per_sample_ms = (total_time_sec / n_samples) * 1000.0
        throughput_sps = n_samples / total_time_sec
        peak_mem_mb = peak_mem / (1024 * 1024)

        print(f"\n[Full Pipeline Benchmark] Samples: {n_samples:,} | Latency: {latency_per_sample_ms:.4f} ms/sample | Throughput: {throughput_sps:.1f} samples/s | Peak Mem: {peak_mem_mb:.2f} MB")

        self.assertLess(latency_per_sample_ms, 15.0) # Full hybrid ML pipeline < 15ms / sample
        self.assertLess(peak_mem_mb, 200.0)          # Peak memory < 200 MB


if __name__ == "__main__":
    unittest.main()
