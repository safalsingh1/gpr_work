/**
 * DPW 1D FDTD Yee Update - CUDA Prototype
 * 
 * Demonstrates the 1D update logic that would reside in the auxiliary grid
 * before projection onto the 3D TF/SF boundary faces.
 */
#include <stdio.h>
#include <stdlib.h>

// Simulated gprMax constants
const float dt_dz = (1e-11 / 0.002); 

/**
 * 1D Auxiliary update kernel for the Magnetic Field (Hx)
 * using the surrounding Electric Field (Ey).
 */
__global__ void update_hx_1d(float *Hx_1D, const float *Ey_1D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check, 1D FDTD Yee cell shift
    if (i < N - 1) {
        Hx_1D[i] -= dt_dz * (Ey_1D[i + 1] - Ey_1D[i]);
    }
}

int main() {
    int N = 256;
    size_t size = N * sizeof(float);
    
    float *h_Hx = (float *)malloc(size);
    float *h_Ey = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_Hx[i] = 0.0f;
        h_Ey[i] = 1.0f; // Uniform Ex field
    }

    float *d_Hx, *d_Ey;
    cudaMalloc((void **)&d_Hx, size);
    cudaMalloc((void **)&d_Ey, size);

    cudaMemcpy(d_Hx, h_Hx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ey, h_Ey, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    update_hx_1d<<<blocksPerGrid, threadsPerBlock>>>(d_Hx, d_Ey, N);

    // Synchronize and copy back
    cudaDeviceSynchronize();
    cudaMemcpy(h_Hx, d_Hx, size, cudaMemcpyDeviceToHost);

    // Verify Output
    printf("DPW 1D Execution Complete.\n");
    printf("Initial Hx[0]: 0.000000 => Updated Hx[0]: %f\n", h_Hx[0]);

    // Cleanup
    cudaFree(d_Hx);
    cudaFree(d_Ey);
    free(h_Hx);
    free(h_Ey);

    return 0;
}
