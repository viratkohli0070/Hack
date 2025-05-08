#include <iostream>
#include <chrono>
#include <cuda.h>

#define N 512  // Square matrix size (try 512 or 1024)

__global__ void matrixMulGPU(int *A, int *B, int *C) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void matrixMulCPU(int *A, int *B, int *C) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int size = N * N * sizeof(int);
    int *A, *B, *C_cpu, *C_gpu;
    int *d_A, *d_B, *d_C;

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C_cpu = (int *)malloc(size);
    C_gpu = (int *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    // CPU Timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;

    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;

    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "First 4 results:\n";
    for (int i = 0; i < 4; i++)
        std::cout << "CPU: " << C_cpu[i] << "\tGPU: " << C_gpu[i] << "\n";

    std::cout << "\nExecution Time:\n";
    std::cout << "CPU Time: " << duration_cpu.count() << " ms\n";
    std::cout << "GPU Time: " << duration_gpu.count() << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_cpu); free(C_gpu);

    return 0;
}
