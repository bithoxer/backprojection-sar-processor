/********************************************************************************************************************
*
* Testing Matrix Addition Functionality
*
*********************************************************************************************************************/

#include <array>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;

// CUDA kernel. Function to add the elements of two arrays 
__global__ void vectorAdd(double* a, double* b, double* c, int vector_size) {
    // First thing we have to do is calculate global thread ID (tid) i.e. figure out what "thread number am I?"
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // block ID (starts from 0) * thread block size + offset thread ID (starts from 0) 

    // Vector boundary guard to ensure we are not accessing out of bounds memory when the vector isn't a multiple of 32 and won't line up properly .
    // Just making sure we don't have any threads with tid greater than vector size attempting to do something because they'll index out of bounds.
    if (tid < vector_size) {
        // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector of size vector_size to int between 0 - 90
void matrix_init(double* a, int vector_size) {
    for (int i=0; i<vector_size; i++) {
        a[i] = rand() % 100;
    }
}

// Check vector add result
void error_check(double* a, double* b, double* c, int vector_size) {
    for (int i=0; i<vector_size; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

/**
 * Read SAR data from given file.
 *
 * This sum is the arithmetic sum, not some other kind of sum that only
 * mathematicians have heard of.
 *
 * @param values Container whose values are summed.
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
void loadData(string filename)

int main() {
    cout << "==================== BACKPROJECTION ALGORITHM LFM-CW SAR PROCESSING =====================\n" << endl;
    
    cout << "Loading data ..." << endl;
    loadData();

    // // Vector size of 2^16 elements
    // int vector_size = 1 << 16; // faster than squaring stuff

    // // Host vector pointers
    // double *h_a, *h_b, *h_c;

    // // Device vector pointers
    // double *d_a, *d_b, *d_c;

    // size_t bytes = sizeof(double) * vector_size; // datatype size_t is unsigned integral type. It represents the size of any object in bytes and returned by sizeof operator.

    // // Allocate host memory i.e. CPU memory
    // h_a = (double*) malloc(bytes);
    // h_b = (double*) malloc(bytes);
    // h_c = (double*) malloc(bytes);

    // // Allocate device memory i.e. GPU memory. Could have used unified memory which gets migrated between CPU and GPU and avoided having to copy stuff as in the step after this.
    // cudaMalloc(&d_a, bytes);
    // cudaMalloc(&d_b, bytes);
    // cudaMalloc(&d_c, bytes);

    // // Initialise vectors a and b on host with random values between 0 and 99
    // matrix_init(h_a, vector_size);
    // matrix_init(h_b, vector_size);
    
    // // Copy data from CPU memory to GPU memory. They are actually two physically different memories.
    // cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); // (gpu_vector, host_vector, size_of_host_vector, direction_of_copy)
    // cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // // Set thread block size and grid size by just tuning for your specific architecture. There are maximum values - look in documentation of architecture.
    // // Thread block size
    // int NUM_THREADS = 256; // needs to be a multiple of 32 because will be translated into warps of size 32 threads. Single number means treated as one dimensional.

    // // Grid size
    // int NUM_BLOCKS = (int) ceil(vector_size/NUM_THREADS); // want a single thread calculating each element of vector addition i.e. vector_size = num_of_threads*num_of_grids

    // // Launch kernel on default stream without shared memory
    // vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, vector_size); // passing device memory pointers into function

    // // Copy answer vector from device back into host memory
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // // Check result for errors
    // error_check(h_a, h_b, h_c, vector_size);

    // cout << "Completed successfully";
    
    






    // float* x; float* y;

    // Allocate Unified Memory - accessible from CPU or GPU
    // cudaMallocManaged(&x, N*sizeof(float));
    // cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on host
    // for (int i = 0; i<N; i++) {
    //     x[i] = 1.0f;
    //     y[i] = 2.0f;
    // }

    // Run kernel on 1M elements on GPU
    // add<<<1, 1>>>(N,x,y);

    // Make CPU wait for GPU to finish before accessing on host because CUDA calling launches don't block CPU
    // cudaDeviceSynchronize();

    // Free memory
    // cudaFree(x);
    // cudaFree(y);


    return 0;
}