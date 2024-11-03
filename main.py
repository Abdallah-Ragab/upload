import pandas as pd
import platform
import subprocess
import re
import fuzzywuzzy.process as fuzz
import time
import numpy as np
from functools import wraps
import memory_profiler


# Load CPU benchmarks from the provided CSV file
def load_benchmarks(csv_path):
    """Load CPU benchmarks from a CSV file."""
    return pd.read_csv(csv_path)


# Get the current system's CPU model
def get_cpu_model():
    """Fetch CPU model based on the operating system."""
    system = platform.system()
    
    if system == "Windows":
        try:
            cpu_info = subprocess.check_output("wmic cpu get name").decode().strip().split('\n')
            cpu_model = cpu_info[1].strip()
        except Exception as e:
            print(f"Error fetching CPU model on Windows: {e}")
            return None

    elif system == "Linux":
        try:
            cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name'", shell=True).decode().strip()
            return re.search(r'model name\s+:\s+(.+)', cpu_info).group(1)
        except Exception as e:
            print(f"Error fetching CPU model on Linux: {e}")
            return None

    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
            return cpu_info
        except Exception as e:
            print(f"Error fetching CPU model on macOS: {e}")
            return None
            
    else:
        print("Unsupported OS")
        return None

    return cpu_model

# Fuzzy match to find the closest CPU name in the benchmarks database
def get_closest_cpu_match(cpu_name, benchmark_df, cpu_col):
    """Find the closest matching CPU name in the benchmarks list using fuzzy matching."""
    choices = benchmark_df[cpu_col].tolist()
    closest_match, score = fuzz.extractOne(cpu_name, choices)
    return closest_match, score


# Get benchmark score for a CPU
def get_benchmark_score(cpu_name, benchmark_df, cpu_col, score_col):
    """Fetch the benchmark score of a given CPU name from the benchmarks dataframe."""
    score_str = benchmark_df[benchmark_df[cpu_col] == cpu_name][score_col].values[0]
    return int(score_str.replace(",", ""))


# Decorator to record the latency of each frame
frame_latencies = []  # Global list to store latencies
max_memory_usage = 0  # Global variable to store max memory usage

def record_performance_metrics(func):
    """Decorator to record the latency of each frame."""
    result = None
    def run_with_mem_record(func, args, kwargs):
        nonlocal result
        result = func(*args, **kwargs)
        
    @wraps(func)
    def wrapper(*args, **kwargs):
        global max_memory_usage
        start_time = time.time()
        # result = func(*args, **kwargs)
        # max_memory_usage = max(memory_profiler.memory_usage((func, args, kwargs)))  # Record the max memory usage
        max_memory_usage = max(memory_profiler.memory_usage((run_with_mem_record, (func, args, kwargs), {})))  # Record the max memory usage
        end_time = time.time()
        latency = end_time - start_time
        frame_latencies.append(latency)  # Record the latency
        return result
    return wrapper


# Simulate YOLOv8 inference
@record_performance_metrics
def simulate_yolov8_inference():
    """Simulate YOLOv8 inference by sleeping for a fixed amount of time."""
    # time.sleep(0.05)  # Simulate model inference (50ms)
    _list = [i for i in range(1000000)] # simulate memory usage
    # return _list


# Run inference on multiple frames and record latencies
def run_inference(num_frames):
    """Run inference on multiple frames and record latencies."""
    for frame_idx in range(num_frames):
        simulate_yolov8_inference()


# Calculate the average FPS based on recorded latencies
def calculate_average_fps():
    """Calculate average FPS based on recorded latencies."""
    if len(frame_latencies) == 0:
        return 0
    average_latency = np.mean(frame_latencies)
    return 1 / average_latency


# Estimate FPS on Raspberry Pi based on local system performance
def estimate_raspberry_pi_fps(local_fps, scaling_factor):
    """Estimate the FPS of the model on Raspberry Pi using the performance scaling factor."""
    return local_fps / scaling_factor


# Main function to execute the pipeline
def main():
    # Load benchmarks data
    csv_path = 'benchmarks.csv'
    benchmarks = load_benchmarks(csv_path)

    # Column names in the CSV file (adjust if necessary)
    cpu_column = 'CPU Name'
    benchmark_column = 'CPU Mark'

    # Step 1: Get the current system's CPU model
    system_cpu_name = get_cpu_model()
    print(f"Current System CPU: {system_cpu_name}")

    # Step 2: Fuzzy match the closest CPU in the benchmarks
    closest_cpu_name, match_score = get_closest_cpu_match(system_cpu_name, benchmarks, cpu_column)
    print(f"Closest match for system CPU: {closest_cpu_name} (Match score: {match_score})")

    # Step 3: Get the benchmark score for the system CPU and Raspberry Pi
    raspberry_pi_cpu_name = "ARM Cortex-A76 4 Core 2400 MHz"  # Adjust if needed

    system_cpu_score = get_benchmark_score(closest_cpu_name, benchmarks, cpu_column, benchmark_column)
    raspberry_pi_score = get_benchmark_score(raspberry_pi_cpu_name, benchmarks, cpu_column, benchmark_column)

    print(f"System CPU Benchmark Score: {system_cpu_score}")
    print(f"Raspberry Pi Benchmark Score: {raspberry_pi_score}")

    # Step 4: Calculate performance scaling factor
    scaling_factor = system_cpu_score / raspberry_pi_score
    print(f"Estimated performance scaling factor (System vs Raspberry Pi): {scaling_factor:.2f}")

    # Step 5: Run YOLOv8 inference simulation for multiple frames
    num_frames = 5  # Number of frames to simulate
    print(f"\nSimulating {num_frames} frames...\n")
    run_inference(num_frames)

    # Step 6: Calculate the average FPS on the local system
    average_fps = calculate_average_fps()
    print(f"\nAverage FPS on Local Machine: {average_fps:.2f} FPS")
    print(f"Max Memory Usage: {max_memory_usage:.2f} MB")
    

    # Step 7: Estimate FPS on Raspberry Pi
    estimated_raspberry_pi_fps = estimate_raspberry_pi_fps(average_fps, scaling_factor)
    print(f"Estimated YOLOv8 FPS on Raspberry Pi 5: {estimated_raspberry_pi_fps:.2f} FPS")


if __name__ == "__main__":
    main()
