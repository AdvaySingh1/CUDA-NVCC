/**
 * @file vector_addition.c
 * @author Advay Singh (Advay@umich.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * 
 * 
 * 
 */



 /** @brief HOST CODE */
// void vecAdd(float * h_A, float * h_B, float * h_C, int N){
//     for (int i = 0; i < N; ++i) h_C[i] = h_A[i] + h_B[i];
// }

// int main(){
//     // Memory allocation for vectors h_A, h_B, and h_C
//     // I/O for vectors h_A and h_B
//     vecAdd(h_A, h_B, h_C, N);
// }

#include <stdlib.h>

void * something = malloc(sizeof(char));

/** @brief CUDA PROGRAMMING MODEL */
// #include <cuda.h>

__global__ void vecAddKernal(float *A, float *B, float *C, N){
    int i = blockDim * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}


void vecAdd(float * A, float * B, float * C, int N){
    float *d_A, *d_B, *d_C;
    int size = N * sizeof(float);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecAddKernal<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 1) Allocate device memory to A, B, and C (this is done sequentially on the CPU) (CPU -> GPU)
// 2) Kernal launch code: device exectued kernal function (this is converted into PTX by NVCC (JIT)) (GPU)
// 3) Copy the resulting vector from the device memory and free device memory allocations (done sequentially on the CPU) (GPU -> CPU)



/* More cuda error checking code. Define a c macro for efficient use. */

// cudaError_t err=cudaMalloc((void **) &d_A, size);
// if (error !=cudaSuccess) {
//  printf(“%s in %s at line %d\n”, cudaGetErrorString(err),__
// FILE__,__LINE__);
// exit(EXIT_FAILURE);
// }



