"""
Refactored PyCUDA / PyOpenCL Template for 1D FDTD Yee Update

This demonstrates how the C++ kernel from `dpw_update_test.cu` should be 
integrated into the gprMax framework using dynamic string templates to 
support multi-precision ($REAL) and cross-platform compute (CUDA/OpenCL).
"""
from string import Template

# Prototype template to be integrated into `gprMax/cuda_opencl/knl_source_updates.py`

dpw_update_hx_1d = Template("""
    // gprMax Macro providing grid indexing 'i' mapped to CUDA/OpenCL threads
    $CUDA_IDX 
    
    // N is the length of the auxiliary 1D grid. 
    // $REAL resolves dynamically to 'float' or 'double'.
    if (i < N - 1) {
        Hx_1D[i] -= dt_dz * (Ey_1D[i + 1] - Ey_1D[i]);
    }
""")
