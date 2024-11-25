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


/** @brief CUDA PROGRAMMING MODEL */
#include <cuda.h>

// void vecAdd(float * A, float * B, float * C, int N){
//     for (int i = 0; i < N; ++i) d_C[i] = d_A[i] + d_B[i];
// }

// 1) Allocate device memory to A, B, and C (this is done sequentially on the CPU) (CPU -> GPU)
// 2) Kernal launch code: device exectued kernal function (this is converted into PTX by NVCC (JIT)) (GPU)
// 3) Copy the resulting vector from the device memory and free device memory allocations (done sequentially on the CPU) (GPU -> CPU)




