"""
Benchmarking Script: CPU vs. GPU Discrete Plane Wave TF/SF

Usage: python bench_cpu_vs_gpu.py <input_file.in>
"""
import time
import sys
import subprocess

def run_simulation(input_file, solver="cpu"):
    """Runs gprMax with the specified solver."""
    cmd = ["python", "-m", "gprMax", input_file]
    if solver == "gpu":
        cmd.append("-gpu")
        
    start_time = time.time()
    
    # Run the process and capture the output to suppress verbose logs
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"[{solver.upper()}] Failed to run. Is the solver supported?")
        print(stderr.decode('utf-8'))
        return None
        
    return end_time - start_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bench_cpu_vs_gpu.py <input_file.in>")
        sys.exit(1)
        
    model_file = sys.argv[1]
    
    print(f"Benchmarking GSoC Proposal - Plane Wave Source: {model_file}")
    print("-" * 50)
    
    # 1. Run Baseline CPU
    print("Running Baseline CPU Cython Solver...")
    cpu_time = run_simulation(model_file, solver="cpu")
    if cpu_time:
        print(f"CPU Time: {cpu_time:.2f} seconds")
        
    # 2. Run Proposed GPU
    print("\nRunning Target GPU PyCUDA Solver...")
    gpu_time = run_simulation(model_file, solver="gpu")
    if gpu_time:
        print(f"GPU Time: {gpu_time:.2f} seconds")
        
    # 3. Report
    print("-" * 50)
    if cpu_time and gpu_time:
        speedup = cpu_time / gpu_time
        print(f"Result: GPU Achieved a {speedup:.2f}x Speedup!")
    elif cpu_time and not gpu_time:
        print("Result: GPU Solver currently unsupported for Plane Waves (Goal of GSoC).")
