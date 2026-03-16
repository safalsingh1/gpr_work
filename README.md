# GPU Acceleration of Plane Wave Source Formulations (TF/SF) in gprMax

This repository serves as **evidence of preparation work** for a Google Summer of Code (GSoC) proposal to port gprMax's Discrete Plane Wave (DPW) Total-Field/Scattered-Field (TF/SF) formulation to the GPU.

## Purpose

Currently, the gprMax FDTD electromagnetic simulator supports the DPW (TF/SF) formulation only on the CPU. Attempting to use a plane wave source with the CUDA or OpenCL solvers raises an error because the boundary injections are not implemented on the device.

My proposed GSoC project aims to:
1. Port the 1D auxiliary FDTD grid to the GPU.
2. Implement the TF/SF boundary phase injections directly on the GPU memory.
3. Eliminate CPU-GPU memory transfer bottlenecks during the FDTD time loop.

This repository demonstrates my understanding of the gprMax codebase, the DPW algorithm, and my ability to write CUDA and PyCUDA kernels for the Yee update.

## Repository Structure

* `docs/gpu_dpw_architecture.md`: Detailed explanation of the architectural plan and integration points.
* `examples/plane_wave_minimal.in`: A minimal gprMax input file to test the parsing logic.
* `prototype_kernels/dpw_update_test.cu`: Original native CUDA prototype showing the 1D DPW Yee update.
* `prototype_kernels/dpw_1d_template.py`: Refactored PyCUDA string template demonstrating alignment with gprMax dynamic compilation and the `$REAL` multi-precision paradigm.
* `benchmarks/bench_cpu_vs_gpu.py`: Script to measure and compare execution times between solvers.
* `tests/test_plane_wave_cpu_gpu.py`: Automated testing script using `pytest` to compare L2-norm field differences.

## Running the CUDA Prototype

The C++ CUDA prototype isolates the 1D FDTD Yee update used in the auxiliary grid. To compile and run it:

```bash
cd prototype_kernels
nvcc dpw_update_test.cu -o dpw_test
./dpw_test
```

## Exploring gprMax DPW Code

For those reviewing my proposal, here is how this project connects to the main gprMax repository:

* **Parsers:** `gprMax/user_objects/cmds_multiuse.py` contains `DiscretePlaneWaveAngles`, `DiscretePlaneWaveVector`, etc., which parse `#plane_wave_angles`.
* **CPU DPW Solver:** `gprMax/cython/plane_wave.pyx` and `fields_updates_normal.pyx` currently handle the grid.
* **Target Integration:** `gprMax/cuda_opencl/knl_fields_updates.py` and `solvers.py` are where the dynamic PyCUDA/PyOpenCL kernels will be integrated.
