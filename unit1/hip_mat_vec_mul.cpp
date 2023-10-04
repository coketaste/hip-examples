#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#define ROWS 128
#define COLUMNS 128
#define HIP_ASSERT(x) (assert((x) == hipSuccess))

// HIP kernel. Each thread takes care of one element of c

__global__ void vecMul(float *A, float *B, float *C, int r, int c)
{
    // shared memory
    __shared__ float s[ROWS * COLUMNS];
    // Get our global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure we do not go out of bounds
    int i = (int)(tid / c);
    int j = tid % c;
    // multilpy
    if (i < r && j < c)
    {
        s[tid] = (A[i * r + j] * B[j]);
        // printf("\ns[%d]=%f",tid,s[tid]);
    }

    __syncthreads();
    // add numbers in each row
    if (tid % c == 0)
    {
        for (int k = 1; k < c; ++k)
            s[tid] += s[i * r + k];
        C[i] = s[tid];
        printf("\nC[%d]=%f", i, C[i]);
    }
}

int main()
{
    // Host vectors
    float *h_arrayA;       // Matrix Array A
    float *h_vectorB;      // Vector Array B
    float *h_vectorC;      // Result Vector
    float *test_h_vectorC; // Result Vector on host

    // Device vectors
    float *d_arrayA;
    float *d_vectorB;
    float *d_vectorC;

    // Size, in bytes, of each vector
    size_t bytes_A = ROWS * COLUMNS * sizeof(float);
    size_t bytes_B = COLUMNS * sizeof(float);
    size_t bytes_C = ROWS * sizeof(float);

    // Allocate memory on host
    h_arrayA = (float *)malloc(bytes_A);
    h_vectorB = (float *)malloc(bytes_B);
    h_vectorC = (float *)malloc(bytes_C);
    test_h_vectorC = (float *)malloc(bytes_C);
    printf("Finished allocating memory for vectors on the CPU\n");

    // Allocate memory for each vector on GPU
    HIP_ASSERT(hipMalloc(&d_arrayA, bytes_A));
    HIP_ASSERT(hipMalloc(&d_vectorB, bytes_B));
    HIP_ASSERT(hipMalloc(&d_vectorC, bytes_C));
    printf("Finished allocating memory for vectors on the GPU\n");

    int i, j;
    // Initialize arrayA
    for (i = 0; i < ROWS; i++)
        for (j = 0; j < COLUMNS; j++)
        {
            h_arrayA[(i * ROWS) + j] = 1; //(i+j);
        }
    // Initialize vectorB
    for (i = 0; i < COLUMNS; i++)
        h_vectorB[i] = 1; //(i);

    // Initialize vectorC
    for (i = 0; i < ROWS; i++)
    {
        h_vectorC[i] = 0;
        test_h_vectorC[i] = 0;
    }

    // Copy host vectors to device
    HIP_ASSERT(hipMemcpy(d_arrayA, h_arrayA, bytes_A, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_vectorB, h_vectorB, bytes_B, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_vectorC, h_vectorC, bytes_C, hipMemcpyHostToDevice));
    printf("\nFinished copying vectors to the GPU\n");

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 512;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)(ROWS * COLUMNS) / blockSize);

    printf("Launching the  kernel on the GPU\n");
    // Execute the kernel
    hipLaunchKernelGGL(vecMul, dim3(gridSize), dim3(blockSize), 0, 0, d_arrayA, d_vectorB, d_vectorC, ROWS, COLUMNS);
    hipDeviceSynchronize();
    printf("\nFinished executing kernel\n");

    // Copy array back to host
    HIP_ASSERT(hipMemcpy(h_vectorC, d_vectorC, bytes_C, hipMemcpyDeviceToHost));

    // test results
    for (i = 0; i < ROWS; i++)
        for (j = 0; j < COLUMNS; j++)
        {
            test_h_vectorC[i] += h_arrayA[i * ROWS + j] * h_vectorB[j];
        }

    printf("Matrix-Vector Multipication \n");
    for (i = 0; i < ROWS; i++)
        if (abs(h_vectorC[i] - test_h_vectorC[i]) > 1e-5)
            printf("Error at position i %d, Expected: %f, Found: %f \n", i, test_h_vectorC[i], h_vectorC[i]);

    // Release device memory
    HIP_ASSERT(hipFree(d_arrayA));
    HIP_ASSERT(hipFree(d_vectorB));
    HIP_ASSERT(hipFree(d_vectorC));

    // Release host memory
    printf("Releasing CPU memory\n");
    free(h_arrayA);
    free(h_vectorB);
    free(h_vectorC);
    free(test_h_vectorC);

    return 0;
}