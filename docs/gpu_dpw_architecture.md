# GPU DPW TF/SF Architecture Plan

## 1. The Discrete Plane Wave (DPW) Algorithm
The Discrete Plane Wave (DPW) method injects an analytical 1D plane wave into a 3D FDTD grid without introducing numerical dispersion errors. 

It works by maintaining an **auxiliary 1D grid**. This 1D grid is stepped forward in time using the standard 1D Yee algorithm. The values from this continuous 1D grid are then projected onto the 3D grid based on the angle and polarization of the plane wave.

## 2. The TF/SF Boundary Method
The Total-Field/Scattered-Field (TF/SF) method divides the 3D computational domain into two regions:
1.  **Total Field Region (inside the box):** Contains incident fields + scattered fields.
2.  **Scattered Field Region (outside the box):** Contains only scattered fields.

Because the FDTD update equations calculate spatial derivatives using neighboring nodes, calculating updates for nodes *on the boundary* mixes Total and Scattered field values, leading to erroneous reflections. 

To solve this, the incident field (from the 1D auxiliary grid) is either **added to** or **subtracted from** the FDTD update equations specifically at the 6 faces of the TF/SF bounding box.

## 3. GPU Architecture Plan

Currently, gprMax performs these boundary corrections entirely on the CPU via Cython. For GPU execution, transferring the 3D boundary arrays between host (CPU) and device (GPU) memory every time step would cripple performance over the PCIe bus.

**The Solution:** Port the entire pipeline to the GPU.

### Architecture Diagram

```asciidoc
[CPU Host]
  1. Parse .in file (#plane_wave_angles)
  2. Compute 1D to 3D phase mappings (proj_idx arrays)
        |
        V
[GPU Device (PyCUDA / PyOpenCL)]
  Memory Allocations:
    - Hx_1D, Ey_1D (1D Grid arrays)
    - proj_idx_x, proj_idx_y, proj_idx_z (Read-only mappings)
    
  Time Loop:
    |-- Launch 1D Auxiliary Yee Update Kernel (1D Block)
    |-- Launch Standard 3D Electric Field Updates
    |-- Launch TF/SF Electric Injection Kernels (6 Faces, 2D Blocks)
    |-- Launch Standard 3D Magnetic Field Updates
    |-- Launch TF/SF Magnetic Injection Kernels (6 Faces, 2D Blocks)
```

## 4. Integration Points in gprMax Codebase

*   **`gprMax/user_objects/cmds_multiuse.py`**: Remove locks restricting `#plane_wave_*` commands from running with the `"cuda"` or `"opencl"` solvers.
*   **`gprMax/solvers.py` (`GPUSolver`)**: Modify initialization to allocate `pycuda.driver.mem_alloc` arrays for the 1D fields and projection mappings. Queue the kernels in the execution stream.
*   **`gprMax/sources.py` (`DiscretePlaneWaveUser`)**: Expose the projection mappings and initial waveform amplitudes so the GPU solver can access them during initialization.
*   **`gprMax/cuda_opencl/knl_source_updates.py`**: Add the new templated PyCUDA strings for the 1D updates and 2D surface injections using `$REAL` macros for multi-precision support.
