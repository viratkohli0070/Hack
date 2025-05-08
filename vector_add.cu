#include <iostream>
#include <cuda.h>
#include <chrono>

#define N 10000000  // 10 million elements

__global__ void vectorAddGPU(float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void vectorAddCPU(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

int main() {
    float *A, *B, *C_cpu, *C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C_cpu = (float *)malloc(size);
    C_gpu = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // CPU Timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;

    // GPU Timing
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();  // Important!
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;

    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "First 5 Results (CPU vs GPU):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "CPU: " << C_cpu[i] << " \tGPU: " << C_gpu[i] << "\n";
    }

    std::cout << "\nExecution Time:\n";
    std::cout << "CPU Time: " << duration_cpu.count() << " ms\n";
    std::cout << "GPU Time: " << duration_gpu.count() << " ms\n";

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_cpu); free(C_gpu);

    return 0;
}
