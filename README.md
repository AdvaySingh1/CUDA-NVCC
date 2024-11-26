# CUDA-NVCC

Linux NVCC CUDA practice

GPU (Parallel Processing)

barrier synchronization

MPI (Message passing interface), OpenMP, and OpenACC are all languages (compiler and runtime) for parallel processing. Using compiler directives, they compile the code to be partially processed.

CUDA goes on to provide APIs for these which is easier to use.
OpenCL is similar to CUDA.

CUDA (framework/architecture) for heterogeneous computing (computing with multiple processors)

Host: CPU, device: GPUs

Thread: group of instructions happen sequentially
Grid: all of the threads initiated by a kernel launch.
When a kernel function is launched (just in time part of CUDA, compiled into device CODE (PTX parallel thread execution) the ISA for GPUs), then a grid is created for the kernel function’s threads.

- Kernel function threads take way fewer clk cycles to schedule and execute than threads in normal CPU

Kernels: device code with CUDA syntax. This also includes helper functions and data structures to be done on the device.

￼

NVCC compiler: Nvidia Cuda Compiler

-     CUDA employs JIT compilation to convert PTX (Parallel Thread Execution) code into machine-specific binary code at runtime. This allows CUDA applications to be optimize 	for different GPU architectures dynamically, but it can cause delays during application startup or CUDA context creation

Global Memory: the same as device memory. The DRAM that comes as part of the GPU. This is usually VRAM or SDRAM and is slower but more parallel.

Memory

- Memory APIs from NVCC which the JIT compiler converts to the correct PTX/Assembly code to allocate device/global (also device) memory.
- cudaMalloc(address of ptr, size)
  - Very similar to the standard c run-time library malloc function.
  - Stronger than malloc run-time (needs the memory) (though keeps track of the size)
  - Takes a generic (void **) variable and writes the address of the pointer into the void ** variable.
  - ex. cudaMalloc((void\*\*)&d_A, size);
  - Directly changes the pointer of a and also is able to return if there’s any errors in the malloc process.
- cudaFree(ptr)
  - Pointer to be freed
- cudaMemcpy(dest ptr, src ptr, size, type)
  - Memory data transfer
  - type is between these (predefined constants in NVCC):
    - cudaMemcpyHostToDevice
    - cudaMemcpyDeviceToHost
    - cudaMemcpyDeviceToDevice
  - Doesn’t work for different GPUs on multi GPU heterogenous systems.
- Don’t dereference pointers on device memory in host code.

￼
Kernals

- Each kernel is an instance of SPMB (Single program multiple data. Similar to SIMD but there can be a different instruction for different data blocks).
- This instance is a grid which is made up of blocks which is made up threads. All threads in a grid execute the same kernel code.
- Built in variables
  - blockDim represents the number of threads in each block of the grid.
    - Struct with x, y, and z dimensions (unsigned integers). Each should be a multiple of 32 (2^5)
  - ThreadIdx & BlockIdx.
  - Examine: i = blockIdx.x \* blockDim + threadIdx.x; c[i] = a[i] + b[i];
- Always kernel memory (ptx has no access to the host memory)
- Keywords:
  - **global** (device execute and host call)
  - **device**(device execute and device call)
  - **host**(host execute and host call)
  - **constant** defines variables to be stored in device constant memory
  - There can be both device and host keyword which causes an overload of the function to be created (done by NVCC).
- Automatic i variable on the stack for each thread (does this mean one for the .x, etc.?)
- Launched from the host via execution configuration parameters.
  - kernalName<<< blocksInGrid, threadsInBlock>>> (other params of the kernal)
  - ex: vecAddKernal<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, N);
  - Each parameter is of type dim3 which is a structure with all 3 dimensions (x, y, z)
  - For 1 dim, set your and z to one.
    - ex: vecAddKernal<<<dim3(3, 1, 1), dim3(1, 2,4)>>>(d_A, d_B, d_C, N);
    - The vecKernalAdd does a shortcut version of this
- Thread specifications
  - gridDim ranges from 1 to 2^16
  - blockDim (in total blockDimx _ blockDimy _ blockDimz) ranges form 1to 2^10
- Thread coordination
  - Barrier Synchronization: Global wave which makes sure that all the previous threads have arrived before continuing. Idea: counter before (# threads initially) and at the wave. When they are the same, we’ve attained barrier synchronization.
  - \_\_syncthreads() called by a thread will cause every thread in the block to weight until they are done with that phase of the kernel execution (that instruction)
    - If there are multiple conditions (if-then-else) then there’s different phases some of them have to go through. So either all of them do then or all do else.
    - if \_\_syncthread() then either all of them do if or all of them don’t do anything (this is only if there’s only and if statement no else) in that phase
    - Importance of temporal proximity: so that the threads don’t have to wait for a long time. So first, the block get’s resources for execution of thread. That’s why the blocks can be executed in any order relative to each other.
    - Transparent Scalability: because we only have to weight for all of the threads in a block to have hardware resources to be scheduled, the blocks can be in any order. This allows for certain systems to reuse the resources and save power while others and execute more blocks concurrently, use more resources, and be faster.
  - Resource Assignment: once grid is generated, they are assigned to hardware resources on a block by block bases
    - These hardware resources are assigned by the NVCC runtime API. The hardware resources in question are Streaming Multiprocessor (SMs). Each SM has a limited number of blocks and limited number of threads (total across blocks) it can have since it takes registers and other reduces to track block indices etc. (partially the reason a block should have atleast 32 threads).
    - More on the hardware (section 4.4 H&P 5th Edition):
      - Streaming Multiprocessor is the same as SIMD processor. Has INT unit and FP unit
      - FP32: single point precision float point ALUs.
      - SM’s have shared memory so for barrier synchronization within the blocks. This is the L1 cache.
      - SM components
        - MTIU (multi thread instruction unit which has Instruction fetch using and wrap scheduler
        - SFU’s (special function units which are not as many and not completely parallel)
        - Sis which have FPs and IP (both alus).
        - Shared memory L1 cache.
        - Register file.

* ￼

Questions

Vector Edition
