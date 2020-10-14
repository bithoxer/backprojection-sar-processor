/*******************************************************************************************************
*
* Backprojection SAR Processor for LFM-CW SAR Implemnted in CUDA
* Author: Thomas Gwasira
* Date: 22 September 2020
*
* This is an NVDIA GPU C++ implementation of a Linear Frequency-Modulated
* Continuous-Wave Synthetic Aperture Radar Backprojection image formation routine.
*
********************************************************************************************************/

// Compilation: nvcc main.cu -I "C:\Users\Thomas\Dropbox\Backprojection-SAR-Processor\gpu_cpp\include" -I "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610/include","C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" -L"C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610/lib/x64" -lcufft -lcudart -o main

# include <iostream>
# include <fstream>
# include <math.h>
# include <memory>
# include <string>

# include <cuda_runtime.h>
# include <cufft.h>
# include <helper_functions.h>
# include <helper_cuda.h>

# include "gpu_backprojection.cuh"
# include "parameters.cuh"

// Derived Parameters
int N_RANGE_COMPRESSED_LEFT_HALF = (int) ceil((double) NFFT/(double) 2);
double az_res_scale = 1;                             // values above 1 reduce resolution, but also computation time
double wavelength = ((double) c0)/fc;                           // carrier wave length
double rres = ((double) c0)/(2*bw);                             // slant range resolution
double a_end = 404.1059;                              
double f_adc_over_4_kr = f_adc/(4*kr) ;
double max_range = c0*f_adc_over_4_kr;               // maximum range without aliasing
double az_res = wavelength/(2*azBW)*az_res_scale;    // azimuth resolution
double mean_z = 346.8229;                            // mean altitude for all platform positions [m]
double delrad = (double) (c0*f_adc_over_4_kr)/(double) PI;                  // meters per radian
double delsamp = (double) (delrad*2*PI)/(double) N_RANGE_BINS;           // meters per real sample
double delrsamp = delsamp*((double) N_RANGE_BINS/(double) NFFT);       // meters per interpolated sample 

int main() 
{
    std::cout << "\n===================== BACKPROJECTION ALGORITHM LFM-CW SAR PROCESSING =====================\n" << std::endl;
    
    // cufftDoubleComplex *trans_test = initialiseSmallMatrix(32, 32);
    // cufftDoubleComplex *trans = transposeMatrix(trans_test, 32, 32);
    // printComplexMatrix(trans, 32, 32);

    ////////// SAR Data //////////
    // Load SAR data  
    cufftDoubleComplex *h_data_windowed_padded = loadAndPrepareSARData();

    // Range compress SAR data
    cufftDoubleComplex *range_compressed = rangeCompression(h_data_windowed_padded);
    int range_compressed_width = (N_RANGE_COMPRESSED_LEFT_HALF+1), range_compressed_height = N_PULSES;

    
    ////////// GEOMETRY //////////
    // Allocate memory for geometry data
    int geometry_length = N_PULSES;
    // size_t geometry_mem_size = sizeof(double) * geometry_length;
    double *r, *a, *z; // memory for these allocated within functions

    // Load geometry data
    loadAndPrepareGeometryData(&r, &a, &z, geometry_length, "../input/casie_geometry_transposed.bin");

    ////////// IMAGE FORMATION //////////
    imageFormation(range_compressed, range_compressed_width, range_compressed_height, r, a, z, geometry_length);

    return 0;
}


void imageFormation(cufftDoubleComplex *h_range_compressed, int range_compressed_width, int range_compressed_height, double *r, double *a, double *z, int geometry_length) {
    
    // Move range_compressed data into device memory for easy access during pulse indexing
    cufftDoubleComplex *d_range_compressed;    
    size_t range_compressed_mem_size = sizeof(cufftDoubleComplex) * range_compressed_width * range_compressed_height;
    checkCudaErrors(cudaMalloc((void **)&d_range_compressed, range_compressed_mem_size));
    checkCudaErrors(cudaMemcpy(d_range_compressed, h_range_compressed, range_compressed_mem_size, cudaMemcpyHostToDevice));

    free(h_range_compressed);

    ////////// Defining Pixel Locations //////////
    // Compute maximum ground range
    double mean_z = doubleArrayMean(z, geometry_length); // Not required if only 1 value of altitude is given
    double ground_range = sqrt(pow(max_range,2) - pow(mean_z,2)); // max_range: hyp and z: opp

    // Create x and y arrays?
    double start = (rres + 30), stop = ground_range, step = rres;
    int x_length = ceil((double) (stop - start)/ (double) step);
    double *x = doubleRange(start, stop, step, x_length);
    x = doubleArrayScalarSubtract(x, x_length, (rres/2));

    start = az_res, stop = a[geometry_length-1], step = az_res;
    int y_length = ceil((double) (stop - start)/ (double) step);
    double *y = doubleRange(start, stop, step, y_length);
    y = doubleArrayScalarSubtract(y, y_length, (az_res/2));
    
    // Create meshgrid of x and y
    int grid_length = x_length * y_length;
    size_t distance_grid_mem_size = sizeof(double) * grid_length;
    
    double *h_X = doubleMeshgridX(x, x_length, y_length);
    double *h_Y = doubleMeshgridY(y, x_length, y_length);

    // Allocate device memory for meshgrid so that it is not reallocated within for loop
    double *d_X, *d_Y;
    checkCudaErrors(cudaMalloc((void **)&d_X, distance_grid_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_Y, distance_grid_mem_size));

    checkCudaErrors(cudaMemcpy(d_X, h_X, distance_grid_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Y, h_Y, distance_grid_mem_size, cudaMemcpyHostToDevice));


    ////////// Initialising Image Grid //////////
    double *z2 = doubleArrayElementwiseSquare(z, geometry_length);

    // Initialise image and phase matrices
    cufftDoubleComplex *d_img;// = complexZeros(x_length, y_length);
    size_t img_mem_size = sizeof(cufftDoubleComplex) * x_length * y_length;
    
    // Allocate memory for image
    checkCudaErrors(cudaMalloc((void **)&d_img, img_mem_size));
    checkCudaErrors(cudaMemset(d_img, 0, img_mem_size));   

    // Constants required for phase computation
    double betapic = 4*PI*kr/(pow(c0,2));
    double lambda4pi = 4*PI/wavelength;

    // Setup execution parameters
    int distance_kernel_threads = 256; // needs to be a multiple of 32 
    int distance_kernel_blocks = (int) ceil((double) grid_length/(double) distance_kernel_threads); // want a single thread calculating for each element of vector



    ////////// Backprojection Loop //////////
    for (int n = 0; n < N_PULSES; n++) {
        std::cout << n << std::endl;

        // Launch kernel on default stream without shared memory to compute distance from antenna
        backprojectionKernel<<<distance_kernel_blocks, distance_kernel_threads>>>(d_img, d_range_compressed, range_compressed_width, range_compressed_height, d_X, r[n], d_Y, a[n], z2[n], betapic, lambda4pi, delrsamp, N_RANGE_COMPRESSED_LEFT_HALF, azBW, grid_length, n);
        
        // Block host until all threads have finished executing
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cufftDoubleComplex *h_img = (cufftDoubleComplex*) malloc(img_mem_size);
    checkCudaErrors(cudaMemcpy(h_img, d_img, img_mem_size, cudaMemcpyDeviceToHost));
    
    writeData(h_img, grid_length, "../output/image.bin");

    // Clean up memory
    free(h_img);
    checkCudaErrors(cudaFree(d_range_compressed));
    checkCudaErrors(cudaFree(d_img));
}

__global__ void backprojectionKernel(cufftDoubleComplex *d_img, cufftDoubleComplex *d_range_compressed, int range_compressed_width, int range_compressed_height, double *d_X, double r_n, double *d_Y, double a_n, double z2_n, double betapic, double lambda4pi, double delrsamp, int  N_RANGE_COMPRESSED_LEFT_HALF, double azBW, int grid_length, int n) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard
    if (tid < grid_length) {
        // Compute distance from antenna to each pixel
        double tx = d_X[tid] - r_n; // x - x_ant
        double tx2 = pow(tx, 2); // (x - x_ant)^2

        double ty = d_Y[tid] - a_n; // y - y_ant
        double ty2 = pow(ty, 2); // (y - y_ant)^2

        double d2 = tx2 + ty2 + z2_n;
        
        double d = sqrt(d2);


        // Estimate azimuth angle to each pixel
        int tx_sign = (0 < tx) - (tx < 0);
        double mz = sqrt(tx2 + z2_n) * tx_sign; // !!! double check use of z2_n
        
        double az = atan2(ty, mz);

        // Interpolate
        int index = (int) round(d/delrsamp);
        if (index > N_RANGE_COMPRESSED_LEFT_HALF) {
            index = N_RANGE_COMPRESSED_LEFT_HALF;
        }
        
        // Compute expected phase
        double betapic_d2 = d2 * betapic;
        double lambda4pi_d = d * lambda4pi;
        double phi = betapic_d2 - lambda4pi_d;

        cufftDoubleComplex pphase;
        pphase.x = cos(-1 * phi);
        pphase.y = sin(-1 * phi);
        
        // Multiply and accumulate image value
        cufftDoubleComplex rc = d_range_compressed[getIndex(n, index, range_compressed_width, range_compressed_height)];
        int az_ = abs(az)<(azBW/2);
        double az_exp = exp(pow(az, 2) * -30); 

        d_img[tid].x = d_img[tid].x + (((pphase.x * rc.x) - (pphase.y * rc.y)) * az_ * az_exp);
        d_img[tid].y = d_img[tid].y + (((pphase.x * rc.y) + (pphase.y * rc.x)) * az_ * az_exp);
    }
}

/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 cufftDoubleComplex* multiplyPhase(cufftDoubleComplex *h_phase_intermediate, int *h_az_, double *h_az2_scaled_exp, int length) {
    // Allocate device memory for input arrays
    cufftDoubleComplex *d_phase_intermediate;
    int *d_az_;
    double *d_az2_scaled_exp;

    checkCudaErrors(cudaMalloc((void **)&d_phase_intermediate, sizeof(cufftDoubleComplex) * length));
    checkCudaErrors(cudaMalloc((void **)&d_az_, sizeof(int) * length));
    checkCudaErrors(cudaMalloc((void **)&d_az2_scaled_exp, sizeof(double) * length));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_phase_intermediate, h_phase_intermediate, sizeof(cufftDoubleComplex) * length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_az_, h_az_, sizeof(int) * length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_az2_scaled_exp, h_az2_scaled_exp, sizeof(double) * length, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    cufftDoubleComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(cufftDoubleComplex) * length));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    multiplyPhaseKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_phase_intermediate, d_az_, d_az2_scaled_exp, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    cufftDoubleComplex *h_out = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * length);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(cufftDoubleComplex) * length, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_phase_intermediate));
    checkCudaErrors(cudaFree(d_az_));
    checkCudaErrors(cudaFree(d_az2_scaled_exp));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for elementwise multiplication of two arrays using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 *
 * @return void.
 */
__global__ void multiplyPhaseKernel(cufftDoubleComplex *d_out, cufftDoubleComplex *d_phase_intermediate, int *d_az_, double *d_az2_scaled_exp, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid].x = d_phase_intermediate[tid].x * d_az_[tid] * d_az2_scaled_exp[tid];
        d_out[tid].y = d_phase_intermediate[tid].y * d_az_[tid] * d_az2_scaled_exp[tid];
    }
}



/**
 * Multiplies each element of an array of doubles by a double scalar.
 *
 * @param h_in Input array.
 * @param length Length of array.
 * @param scalar Scalar to multiply each array element.
 *
 * @return Array containing products of each input array element and scalar.
 */
int* compareDoubleArrayToThreshold(double* h_in, int length, double threshold) {
    size_t in_mem_size = sizeof(double) * length;
    size_t out_mem_size = sizeof(int) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, in_mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    int *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, out_mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    compareDoubleArrayToThresholdKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, threshold);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    int *h_out = (int*) malloc(out_mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for multiplication of each element of an array and double scalar on GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to multiply each array element.
 *
 * @return void.
 */
__global__ void compareDoubleArrayToThresholdKernel(double *d_in, int *d_out, int length, double threshold) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = (abs(d_in[tid]) < threshold);
    }
}

/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 cufftDoubleComplex* indexRangeCompressedData(cufftDoubleComplex *d_range_compressed, int range_compressed_width, int range_compressed_height, int *h_ids, int ids_length, int n) {
    size_t ids_mem_size = sizeof(int) * ids_length;
    size_t out_mem_size = sizeof(cufftDoubleComplex) * ids_length;

    // Allocate device memory for input arrays
    int *d_ids;
    checkCudaErrors(cudaMalloc((void **)&d_ids, ids_mem_size));

    // Copy host data to device
    checkCudaErrors(cudaMemcpy(d_ids, h_ids, ids_mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    cufftDoubleComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, out_mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) ids_length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    indexRangeCompressedDataKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_range_compressed, range_compressed_width, range_compressed_height, d_ids, ids_length, n);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    cufftDoubleComplex *h_out = (cufftDoubleComplex*) malloc(out_mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_ids));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for elementwise multiplication of two arrays using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 * @param k Pulse index
 *
 * @return void.
 */
__global__ void indexRangeCompressedDataKernel(cufftDoubleComplex *d_out, cufftDoubleComplex *d_range_compressed, int range_compressed_width, int range_compressed_height, int *d_ids, int length, int n) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        int range_compressed_pos = getIndex(n, d_ids[tid], range_compressed_width, range_compressed_height);
        d_out[tid].x = d_range_compressed[range_compressed_pos].x;
        d_out[tid].y = d_range_compressed[range_compressed_pos].y;
    }
}






/**
 * Generate matrix of complex zeros with given dimenstions.
 *
 * @param width Width of matrix. 
 * @param height Height of matrix.
 *
 * @return grid of cufftDoubleComplex zeros.
 */
cufftDoubleComplex* complexZeros(int width, int height) {
    size_t mem_size = sizeof(cufftDoubleComplex) * width * height;
    
    // Allocate memory for output matrix
    cufftDoubleComplex *mtx_out = (cufftDoubleComplex*) malloc(mem_size);
    memset(mtx_out, 0, mem_size);

    return mtx_out;
}


//------------------------------------------------------------------------------------------------------
// ALGORITHM METHODS
//------------------------------------------------------------------------------------------------------
cufftDoubleComplex* loadAndPrepareSARData(void) {
    
    ////////// Read SAR data from file //////////
    // Allocate host memory for SAR data
    cufftDoubleComplex *h_data; // host data
    size_t data_length = N_PULSES * N_RANGE_BINS;
    size_t h_data_mem_size = sizeof(cufftDoubleComplex) * data_length; // size of dataset
    h_data = (cufftDoubleComplex*) malloc(h_data_mem_size);

    // Read data
    loadDataIntoComplex(h_data, data_length, "../input/casie_raw_data.bin");


    ////////// Apply window to SAR data //////////
    // Generate window function length R
    cufftDoubleComplex* h_window_matrix = blackmanMatrix(N_RANGE_BINS, N_PULSES); // recall data read as N stacks of R down-range values
    // writeData(h_window_matrix, data_length, "../output/blackman_matrix.bin");

    // Allocate device memory for SAR data
    cufftDoubleComplex *d_data; // device data
    size_t d_data_mem_size = sizeof(cufftDoubleComplex) * data_length; // ensure sizes correspond with size of signals R should equal BATCH?
    checkCudaErrors(cudaMalloc((void **)&d_data, d_data_mem_size));

    // Copy host memory for data to device
    checkCudaErrors(cudaMemcpy(d_data, h_data, d_data_mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for window
    cufftDoubleComplex *d_window_matrix; // device data
    size_t d_window_matrix_mem_size = sizeof(cufftDoubleComplex) * N_PULSES * N_RANGE_BINS;
    checkCudaErrors(cudaMalloc((void **)&d_window_matrix, d_window_matrix_mem_size));

    // Copy host memory for window to device
    checkCudaErrors(cudaMemcpy(d_window_matrix, h_window_matrix, d_window_matrix_mem_size, cudaMemcpyHostToDevice));
    
    // Allocate host memory for windowed data
    cufftDoubleComplex *h_data_windowed; // host data
    h_data_windowed = (cufftDoubleComplex*) malloc(h_data_mem_size);

    // Allocate device memory for answer
    cufftDoubleComplex *d_data_windowed; // device windowed data
    checkCudaErrors(cudaMalloc((void **)&d_data_windowed, d_data_mem_size));

    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) data_length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    elementWiseComplexMultiplyKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_data, d_window_matrix, d_data_windowed, data_length);

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy windowed data from device to host memory
    checkCudaErrors(cudaMemcpy(h_data_windowed, d_data_windowed, d_data_mem_size, cudaMemcpyDeviceToHost));
/*    writeData(h_data_windowed, data_length, "../output/windowed_unpadded_data.bin");*/


    ////////// Pad windowed data //////////
    cufftDoubleComplex *h_data_windowed_padded;
    zeroPadMatrixRows(h_data_windowed, &h_data_windowed_padded, N_RANGE_BINS, NFFT, N_PULSES);
    
    // Memory clean up
    // free(h_data);
    free(h_window_matrix);
    free(h_data_windowed);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_window_matrix));
    checkCudaErrors(cudaFree(d_data_windowed));

    return h_data_windowed_padded;
}    
    

cufftDoubleComplex* rangeCompression(cufftDoubleComplex* h_data_windowed_padded) {   
    // Allocate memory for range compressed data
    cufftDoubleComplex *range_compressed;
    range_compressed = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * NFFT * BATCH);
    
    //  Transform signal
    std::cout << "Performing 1-D FFT on " << BATCH << " x " << NFFT << " matrix." << std::endl;
    for (int i=0; i<BATCH_SPLIT; i++) {
        // Allocate device memory for data to FFT
        cufftDoubleComplex *d_data_windowed_padded; // device data
        size_t d_data_mem_size = sizeof(cufftDoubleComplex) * NFFT * BATCH/BATCH_SPLIT; // ensure sizes correspond with size of signals R should equal BATCH?
        checkCudaErrors(cudaMalloc((void **)&d_data_windowed_padded, d_data_mem_size));

        // Copy host memory to device
        checkCudaErrors(cudaMemcpy(d_data_windowed_padded, (h_data_windowed_padded + i * NFFT * BATCH/BATCH_SPLIT), d_data_mem_size, cudaMemcpyHostToDevice));
            
        // cuFFT Plan
        cufftHandle plan;
        checkCudaErrors(cufftPlan1d(&plan, NFFT, CUFFT_Z2Z, BATCH/BATCH_SPLIT)); 
        checkCudaErrors(cufftExecZ2Z(plan, (cufftDoubleComplex *) d_data_windowed_padded, (cufftDoubleComplex *) d_data_windowed_padded, CUFFT_FORWARD)); // using double precision version of Exec
    
        // Copy transformed data back to host memory
        checkCudaErrors(cudaMemcpy((range_compressed + i * NFFT * BATCH/BATCH_SPLIT), d_data_windowed_padded, d_data_mem_size, cudaMemcpyDeviceToHost));

        // Block host until all threads have finished executing
        checkCudaErrors(cudaDeviceSynchronize());

        // Clean up memory
        checkCudaErrors(cudaFree(d_data_windowed_padded));

        checkCudaErrors(cufftDestroy(plan));
    }
    
    std::cout << "FFT performed successfully.\n" << std::endl;
    
    // Discard right half of fft result
    cufftDoubleComplex *range_compressed_left_half;
    discardLastMatrixRows(range_compressed, &range_compressed_left_half, NFFT, ceil(N_RANGE_COMPRESSED_LEFT_HALF), BATCH);

    // Zeropad left half of fft result
    cufftDoubleComplex *range_compressed_left_half_padded;
    zeroPadMatrixRows(range_compressed_left_half, &range_compressed_left_half_padded, ceil(N_RANGE_COMPRESSED_LEFT_HALF), (ceil(N_RANGE_COMPRESSED_LEFT_HALF) + 1), BATCH);

    // Clean up memory
    free(range_compressed);
    free(range_compressed_left_half);

    // writeData(range_compressed_left_half_padded, ((ceil(N_RANGE_COMPRESSED_LEFT_HALF) + 1) * BATCH), "../output/range_compressed.bin");
    return range_compressed_left_half_padded;
}

/**
 * Loads geometry data from given filename.
 * Assumes data file contains 4 sequential chunks of geometry data: time, latitude, longitude, altitude all of which
 * have length = no. of pulses.
 *
 * @param h_time Time data.
 * @param h_latitude Latitude data.
 * @param h_longitude Longitude data.
 * @param h_z Altitude data.
 *
 * @return void.
 */
void loadAndPrepareGeometryData(double **r, double **a, double **z, int geometry_length, std::string filename) {

    ////////// Read geometry data from file into corresponding memory locations //////////
    // Allocate host memory for all geometry data
    double *h_geometry_data; // all geometry data
    size_t all_geometry_data_length = N_GEOMETRY * geometry_length;
    size_t h_geometry_data_mem_size = sizeof(double) * all_geometry_data_length;
    h_geometry_data = (double*) malloc(h_geometry_data_mem_size);

    // Read data
    loadDataIntoDouble(h_geometry_data, all_geometry_data_length, filename);

    // Allocate memory for separated geometry data
    size_t geometry_mem_size = sizeof(double) * geometry_length;
    double *time = (double*) malloc(geometry_mem_size),
    *latitude= (double*) malloc(geometry_mem_size),
    *longitude = (double*) malloc(geometry_mem_size),
    *altitude = (double*) malloc(geometry_mem_size);
    
    // Separate the different types of datasets into corresponding arrays
    memcpy(time, h_geometry_data, sizeof(double) * geometry_length); // assuming N_PULSES = width of geometry data. To-do: Make general.
    memcpy(latitude, (h_geometry_data + (geometry_length * 1)), sizeof(double) * geometry_length);
    memcpy(longitude, (h_geometry_data + (geometry_length * 2)), sizeof(double) * geometry_length);
    memcpy(altitude, (h_geometry_data + (geometry_length * 3)), sizeof(double) * geometry_length);

    
    ////////// Convert GPS coordinate system to UTM coordinate system //////////
    double mean_lat = doubleArrayMean(latitude, geometry_length); // reference latitude
    
    // Conversion factors
    double lat_cf = 1852.23 * 60.0; // latitude conversion factor to convert lat to northing
    double long_cf = lat_cf*cos(mean_lat*PI/180);  // longitude conversion factor to convert longitude to easting
    
    // Convert arrays of latitude and longitude
    double *northing = doubleArrayScalarMultiply(latitude, geometry_length, lat_cf);
    double *easting = doubleArrayScalarMultiply(longitude, geometry_length, long_cf);

    // // Subtract cable delay from altitude
    double *h_z = doubleArrayScalarSubtract(altitude, geometry_length, CABLE_DELAY); // altitude by cabledelay;


    // ////////// Convert UTM coordinate system into local UTM coordinate system //////////
    double rot_angle = -atan2((easting[geometry_length-1]-easting[0]),(northing[geometry_length-1]-northing[0])) + PI/2;
    double c_angle = cos(rot_angle);
    double s_angle = sin(rot_angle);
    double *l_northing = doubleArrayScalarSubtract(northing, geometry_length, northing[0]);
    double *l_easting = doubleArrayScalarSubtract(easting, geometry_length, easting[0]);

    
    // SAR Antenna Phase Center
    double *h_r = doubleArrayVectorSubtract((doubleArrayScalarMultiply(l_northing, geometry_length, c_angle)),
                                    (doubleArrayScalarMultiply(l_easting, geometry_length, s_angle)),
                                    geometry_length); // x[n]
    
    double *h_a = doubleArrayVectorAdd((doubleArrayScalarMultiply(l_northing, geometry_length, s_angle)), 
                                (doubleArrayScalarMultiply(l_easting, geometry_length, c_angle)),
                                geometry_length);  // y[n]

    *r = h_r;
    *a = h_a;
    *z = h_z;    

    // Clean up memory
    free(h_geometry_data);
    free(time);
    free(latitude);
    free(longitude);
    free(altitude);
    free(northing);
    free(easting);
    free(l_northing);
    free(l_easting);
}

/**
 * Generates complex phase matrix from given phase term.
 *
 * @param h_phi Phase terms.
 * @param length Length of array.
 *
 * @return Array containing complex phase terms.
 */
 cufftDoubleComplex* expectedPhase(double *h_in, int length) {
    size_t in_mem_size = sizeof(double) * length;
    size_t out_mem_size = sizeof(cufftDoubleComplex) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, in_mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    cufftDoubleComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, out_mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    expectedPhaseKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    cufftDoubleComplex *h_out = (cufftDoubleComplex*) malloc(out_mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for subtraction of double scalar from each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return void.
 */
__global__ void expectedPhaseKernel(double *d_in, cufftDoubleComplex *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid].x = cos(-1 * d_in[tid]);
        d_out[tid].y = sin(-1 * d_in[tid]);
    }    
}

//------------------------------------------------------------------------------------------------------
// DOUBLE ARRAY/MATRIX OPERATIONS
//------------------------------------------------------------------------------------------------------
/**
 * Determines the mean of all values in an array of doubles.
 *
 * @param arr Array containing values for which mean should be determined.
 * @param length Length of array.
 *
 * @return mean of values in array.
 */
double doubleArrayMean(double* h_in, int length) {
    double sum = 0;
    for (int i=0; i<length; i++) {
        sum += h_in[i];
    }

    return sum/length;
}

/**
 * Multiplies each element of an array of doubles by a double scalar.
 *
 * @param h_in Input array.
 * @param length Length of array.
 * @param scalar Scalar to multiply each array element.
 *
 * @return Array containing products of each input array element and scalar.
 */
double* doubleArrayScalarMultiply(double* h_in, int length, double scalar) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayScalarMultiplyKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, scalar);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for multiplication of each element of an array and double scalar on GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to multiply each array element.
 *
 * @return void.
 */
__global__ void doubleArrayScalarMultiplyKernel(double *d_in, double *d_out, int length, double scalar) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in[tid] * scalar;
    }
}

/**
 * Divides each element of an array of doubles by scalar.
 *
 * @param h_in Input array.
 * @param length Length of array.
 * @param scalar Scalar to be divide elements of array.
 *
 * @return Array containing integer quotient of each input array element and scalar.
 */
 int* doubleArrayScalarDivide(double *h_in, int length, double scalar) {
    size_t in_mem_size = sizeof(double) * length;
    size_t out_mem_size = sizeof(int) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, in_mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    int *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, out_mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayScalarDivideKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, scalar);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    int *h_out = (int*) malloc(out_mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for dividion of each element of an array by scalar using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be added to each array element.
 *
 * @return void.
 */
__global__ void doubleArrayScalarDivideKernel(double *d_in, int *d_out, int length, double scalar) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = (int) round(d_in[tid]/scalar);
    }
}

/**
 * Adds a double scalar to each element of an array of doubles.
 *
 * @param h_in Input array.
 * @param length Length of array.
 * @param scalar Scalar to be added to each array element.
 *
 * @return Array containing difference between of each input array element and scalar.
 */
 double* doubleArrayScalarAdd(double *h_in, int length, double scalar) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayScalarAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, scalar);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for addition of double scalar to each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be added to each array element.
 *
 * @return void.
 */
__global__ void doubleArrayScalarAddKernel(double *d_in, double *d_out, int length, double scalar) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in[tid] + scalar;
    }
}





/**
 * Subtracts a double scalar from each element of an array of doubles.
 *
 * @param h_in Input array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return Array containing difference between of each input array element and scalar.
 */
double* doubleArrayScalarSubtract(double *h_in, int length, double scalar) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayScalarSubtractKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, scalar);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for subtraction of double scalar from each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return void.
 */
__global__ void doubleArrayScalarSubtractKernel(double *d_in, double *d_out, int length, double scalar) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in[tid] - scalar;
    }
}

/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 double* doubleArrayVectorElementwiseMultiply(double *h_in_a, double *h_in_b, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input arrays
    double *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayVectorElementwiseMultiplyKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for elementwise multiplication of two arrays using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 *
 * @return void.
 */
__global__ void doubleArrayVectorElementwiseMultiplyKernel(double *d_in_a, double *d_in_b, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in_a[tid] * d_in_b[tid];
    }
}

/**
 * Adds two double arrays.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
double* doubleArrayVectorAdd(double *h_in_a, double *h_in_b, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input arrays
    double *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayVectorAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for vector addition of two arrays using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 *
 * @return void.
 */
__global__ void doubleArrayVectorAddKernel(double *d_in_a, double *d_in_b, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in_a[tid] + d_in_b[tid];
    }
}

/**
 * Subtracts a double array from another.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing difference between input arrays.
 */
double* doubleArrayVectorSubtract(double *h_in_a, double *h_in_b, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input arrays
    double *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayVectorSubtractKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for vector subtraction of one array from another using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 *
 * @return void.
 */
__global__ void doubleArrayVectorSubtractKernel(double *d_in_a, double *d_in_b, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = d_in_a[tid] - d_in_b[tid];
    }
}

/**
 * Generate array of elements between specified values with specified space between them.
 *
 * @param start Begining value in generated array.
 * @param stop Ending value in generated array.
 * @param step Step size between two consecutive elements. Defaults to 1.
 *
 * @return array containing equally spaced elements withing given range.
 */
double* doubleRange(double start, double stop, double step, int length) {
    // Allocate memory for output array
    size_t mem_size = sizeof(double) * length;
    double *arr_out = (double*) malloc(mem_size);

    int i = 0;
    for (double value = start; value < stop; value += step) {
        arr_out[i] = value;
        i++;
    }

    return arr_out;
}

/**
 * Squares each element of an array of doubles.
 *
 * @param h_in Input array.
 * @param length Length of array.
 *
 * @return Array containing squared values of original array.
 */
 double* doubleArrayElementwiseSquare(double *h_in, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayElementwiseSquareKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for subtraction of double scalar from each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return void.
 */
__global__ void doubleArrayElementwiseSquareKernel(double *d_in, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = pow(d_in[tid], 2);
    }    
}

/**
 * Obtains square root of each element of an array of doubles.
 *
 * @param h_in Input array.
 * @param length Length of array.
 *
 * @return Array containing square root values of original array.
 */
 double* doubleArrayElementwiseSquareRoot(double *h_in, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayElementwiseSquareRootKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for subtraction of double scalar from each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return void.
 */
__global__ void doubleArrayElementwiseSquareRootKernel(double *d_in, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = sqrt(d_in[tid]);
    }    
}

/**
 * Generate matrix of coordinates given by x and y arrays were x forms each row's x coordinates and y forms each row's y coordinates.
 *
 * @param x Array whose values form each row of output matrix.
 * @param y Array whose values form each column of output matrix.
 * @param x_length Length of x array and therefore width of matrix.
 * @param y_length Length of y array and therefore height of matrix.
 * x length = y length. Both passed as parameters to facilitate a length match check.
 *
 * @return matrix containing x,y coordinates.
 */
 double2* doubleMeshgrid(double *x, double *y, int x_length, int y_length) {
    size_t mem_size = sizeof(double2) * x_length * y_length;
    double2 *mtx_out = (double2*) malloc(mem_size);

    int row, col;
    for (row = 0; row < y_length; row++) {
        for (col = 0; col < x_length; col++) {
            int index = getIndex(row, col, x_length, y_length);
            mtx_out[index].x = x[col];
            mtx_out[index].y = y[row]; 
        }
    }

    std::cout << "height = " << row << "width = " << col << std::endl;
    return mtx_out;
}

/**
 * Generate matrix of x coordinates from a given array.
 *
 * @param x Array whose values form each row of output matrix.
 * @param x_length Length of x array and therefore width of matrix.
 * @param y_length Length of y array and therefore height of matrix.
 * x length = y length. Both passed as parameters to facilitate a length match check.
 *
 * @return matrix containing x coordinates.
 */
 double* doubleMeshgridX(double *x, int x_length, int y_length) {
    size_t mem_size = sizeof(double) * x_length * y_length;
    double *arr_out = (double*) malloc(mem_size);

    for (int row = 0; row < y_length; row++) {
        for (int col = 0; col < x_length; col++) {
            int index = getIndex(row, col, x_length, y_length);
            arr_out[index] = x[col];
        }
    }

    return arr_out;
}

/**
 * Generate matrix of y coordinates from a given array.
 *
 * @param y Array whose values form each column of output matrix.
 * @param x_length Length of x array and therefore width of matrix.
 * @param y_length Length of y array and therefore height of matrix.
 * x length = y length. Both passed as parameters to facilitate a length match check.
 *
 * @return array containing y coordinates.
 */
 double* doubleMeshgridY(double *y, int x_length, int y_length) {
    size_t mem_size = sizeof(double) * x_length * y_length;
    double *arr_out = (double*) malloc(mem_size);

    for (int row = 0; row < y_length; row++) {
        for (int col = 0; col < x_length; col++) {
            int index = getIndex(row, col, x_length, y_length);
            arr_out[index] = y[row];
        }
    }

    return arr_out;
}


/**
 * Gives indication of sign of each element in array of doubles
 *
 * @param h_in Input array.
 * @param length Length of array.
 *
 * @return Array containing indications of signs of each element in original array.
 */
 double* doubleArraySign(double* h_in, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArraySignKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel for determining sign of each element of an array on GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 *
 * @return void.
 */
 __global__ void doubleArraySignKernel(double *d_in, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = (0 < d_in[tid]) - (d_in[tid] < 0);
    }
}

/**
 * Sets any indexes out of bounds to last possible value.
 *
 * @param h_in Input array.
 * @param length Length of array.
 *
 * @return Array containing boundary corrected indexes.
 */
 int* boundaryCorrectIndexes(int* h_in, int length, int N) {
    size_t mem_size = sizeof(int) * length;

    // Allocate device memory for input array
    int *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    int *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    boundaryCorrectIndexesKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length, N);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    int *h_out = (int*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel for correcting boundary values of indexes.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 *
 * @return void.
 */
 __global__ void boundaryCorrectIndexesKernel(int *d_in, int *d_out, int length, int N) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        if (d_in[tid] > N) {
            d_out[tid] = N;
        }

        else {
            d_out[tid] = d_in[tid]; 
        }
    }
}

/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 double* doubleArrayVectorAtan2(double *h_in_a, double *h_in_b, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input arrays
    double *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayVectorAtan2Kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for elementwise multiplication of two arrays using GPU.
 *
 * @param d_in_a @param d_in_b Input arrays in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of arrays.
 *
 * @return void.
 */
__global__ void doubleArrayVectorAtan2Kernel(double *d_in_a, double *d_in_b, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = atan2(d_in_a[tid], d_in_b[tid]);
    }
}

/**
 * Obtains square root of each element of an array of doubles.
 *
 * @param h_in Input array.
 * @param length Length of array.
 *
 * @return Array containing square root values of original array.
 */
 double* doubleArrayElementwiseExp(double *h_in, int length) {
    size_t mem_size = sizeof(double) * length;

    // Allocate device memory for input array
    double *d_in;
    checkCudaErrors(cudaMalloc((void **)&d_in, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    double *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    doubleArrayElementwiseExpKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    double *h_out = (double*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}

/**
 * Kernel function for subtraction of double scalar from each element of an array using GPU.
 *
 * @param d_in Input array in device memory.
 * @param d_out Device memory location for output array.
 * @param length Length of array.
 * @param scalar Scalar to be subtracted from each array element.
 *
 * @return void.
 */
__global__ void doubleArrayElementwiseExpKernel(double *d_in, double *d_out, int length) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    if (tid < length) {
        d_out[tid] = exp(d_in[tid]);
    }    
}

//------------------------------------------------------------------------------------------------------
// CUFFTDOUBLECOMPLEX MATRIX OPERATIONS
//------------------------------------------------------------------------------------------------------
/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 cufftDoubleComplex* elementWiseComplexAdd(cufftDoubleComplex *h_in_a, cufftDoubleComplex *h_in_b, int length) {
    size_t mem_size = sizeof(cufftDoubleComplex) * length;

    // Allocate device memory for input arrays
    cufftDoubleComplex *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    cufftDoubleComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    elementWiseComplexAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    cufftDoubleComplex *h_out = (cufftDoubleComplex*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}


/**
 * Kernel function for element-wise multiplication of two cufftDoubleComplex matrices into the first argument
 */
 __global__ void elementWiseComplexAddKernel(cufftDoubleComplex* d_in_a, cufftDoubleComplex* d_in_b, cufftDoubleComplex* d_out, int length) {
    // Calculate global thread ID (tid) i.e. figure out what "thread number am I?"
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // block ID (starts from 0) * thread block size + offset thread ID (starts from 0) 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    // i.e. just making sure no threads with tid greater than vector size attempting to do something because they'll index out of bounds.
    if (tid < length) {
        d_out[tid].x = (d_in_a[tid].x + d_in_b[tid].x);
        d_out[tid].y = (d_in_a[tid].y + d_in_b[tid].y);
    }
}



/**
 * Multiplies two double arrays elementwise.
 *
 * @param h_in_a @param h_in_b Input arrays.
 * @param length Length of each array.
 *
 * @return Array containing sum of input arrays.
 */
 cufftDoubleComplex* elementWiseComplexMultiply(cufftDoubleComplex *h_in_a, cufftDoubleComplex *h_in_b, int length) {
    size_t mem_size = sizeof(cufftDoubleComplex) * length;

    // Allocate device memory for input arrays
    cufftDoubleComplex *d_in_a, *d_in_b;
    checkCudaErrors(cudaMalloc((void **)&d_in_a, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_in_b, mem_size));

    // Copy h_in to d_in
    checkCudaErrors(cudaMemcpy(d_in_a, h_in_a, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_b, h_in_b, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output array
    cufftDoubleComplex *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, mem_size));
    
    // Setup execution parameters
    int NUM_THREADS = 256; // needs to be a multiple of 32 
    int NUM_BLOCKS = (int) ceil((double) length/(double) NUM_THREADS); // want a single thread calculating for each element of vector
    // Launch kernel on default stream without shared memory
    elementWiseComplexMultiplyKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_in_a, d_in_b, d_out, length);

    // Block host until all threads have finished executing
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for output array
    cufftDoubleComplex *h_out = (cufftDoubleComplex*) malloc(mem_size);

    // Copy d_out to h_out
    checkCudaErrors(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_in_a));
    checkCudaErrors(cudaFree(d_in_b));
    checkCudaErrors(cudaFree(d_out));

    return h_out;    
}


/**
 * Kernel function for element-wise multiplication of two cufftDoubleComplex matrices into the first argument
 */
 __global__ void elementWiseComplexMultiplyKernel(cufftDoubleComplex* d_in_a, cufftDoubleComplex* d_in_b, cufftDoubleComplex* d_out, int length) {
    // Calculate global thread ID (tid) i.e. figure out what "thread number am I?"
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // block ID (starts from 0) * thread block size + offset thread ID (starts from 0) 

    // Vector boundary guard to ensure not accessing out of bounds memory when the vector isn't a multiple of 32
    // i.e. just making sure no threads with tid greater than vector size attempting to do something because they'll index out of bounds.
    if (tid < length) {
        d_out[tid].x = (d_in_a[tid].x * d_in_b[tid].x) - (d_in_a[tid].y * d_in_b[tid].y);
        d_out[tid].y = (d_in_a[tid].x * d_in_b[tid].y) + (d_in_a[tid].y * d_in_b[tid].x);
    }
}

cufftDoubleComplex* transposeMatrix(const cufftDoubleComplex* h_idata, const unsigned int width, const unsigned int height) {
    // Allocate device memory
    cufftDoubleComplex* d_idata;
    cufftDoubleComplex* d_odata;
    const unsigned int mem_size = sizeof(cufftDoubleComplex) * width * height;
    checkCudaErrors(cudaMalloc((void**) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc( (void**) &d_odata, mem_size));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice));


    // Setup execution parameters
    // dim3 grid(width / BLOCK_DIM, height / BLOCK_DIM, 1);
    dim3 grid((width + (BLOCK_DIM-(width%BLOCK_DIM)))/BLOCK_DIM, (height + (BLOCK_DIM-(height%BLOCK_DIM)))/BLOCK_DIM);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    transposeMatrixKernel<<< grid, threads >>>(d_odata, d_idata, width, height);
    

	// Synchronize threads
	cudaThreadSynchronize();

    // Copy result from device to host
    cufftDoubleComplex* h_odata = (cufftDoubleComplex*) malloc(mem_size);
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost));

    // Clean up memory
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    // cudaThreadExit();

    return h_odata;
}

/**
 * Kernel for coalesced matrix transpose.
 * This kernel is optimized to ensure all global reads and writes are coalesced,
 * and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
 * than the naive kernel below.  Note that the shared memory array is sized to 
 * (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
 * so that bank conflicts do not occur when threads address the array column-wise.
 *
 * Credit: https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
 *
 * @param odata Output data location in device memory.
 * @param idata Input data in device memory.
 * @param width Width of input matrix.
 * @param height Height of input matrix.
 *
 * @return void.
 */
__global__ void transposeMatrixKernel(cufftDoubleComplex *odata, cufftDoubleComplex *idata, int width, int height)
{
	__shared__ cufftDoubleComplex block[BLOCK_DIM][BLOCK_DIM+1];
    
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    unsigned int index_in = xIndex + (yIndex)*width;
    unsigned int index_out = xIndex + (yIndex)*height;

    if((xIndex < width) && (yIndex < height)) {
        for (int i=0; i<BLOCK_DIM; i+=BLOCK_ROWS) {
            block[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }
    }
    __syncthreads();

    if((xIndex < width) && (yIndex < height)) {
        for (int i=0; i<BLOCK_DIM; i+=BLOCK_ROWS) {
            odata[index_out+i*height] = block[threadIdx.x][threadIdx.y+i];
        }
    }
}

/**
 * Trims matrix rows (from right) to specified new width.
 */
 void discardLastMatrixRows(const cufftDoubleComplex *data, cufftDoubleComplex **trimmed_data, int old_width, int new_width, int height) {
    // Allocate host memory for new padded data
    cufftDoubleComplex *new_data = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * new_width * height);

    for (int x=0; x<height; x++) {  // remember, column length is number of rows
        int row_beg = getIndex(x, 0, new_width, height);
        
        memcpy(new_data + row_beg, data + getIndex(x, 0, old_width, height), sizeof(cufftDoubleComplex) * new_width);
    }

    *trimmed_data = new_data;
}



/**
 * Zero pads all rows of given flat matrix to a specified length
 *
 * @param data cufftDoubleComplex pointer to the data to be zero padded.
 * @param padded_data Pointer to a cufftDoubleComplex pointer of the location of the new padded data.
 * @param old_data_length Length of data to be padded.
 * @param new_data_length Desired length of data.
 *
 * @return cufftDoubleComplex pointer to new padded data.
 */
void zeroPadMatrixRows(const cufftDoubleComplex *data, cufftDoubleComplex **padded_data, int old_width, int new_width, int height) {
    // Allocate host memory for new padded data
    cufftDoubleComplex *new_data = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * new_width * height);
    
    for (int x=0; x<height; x++) {  // remember, column length is number of rows
        int row_beg = getIndex(x, 0, new_width, height);
        
        memcpy(new_data + row_beg, data + getIndex(x, 0, old_width, height), sizeof(cufftDoubleComplex) * old_width);
        memset(new_data + row_beg + old_width, 0, sizeof(cufftDoubleComplex)*(new_width -  old_width));
    }

    *padded_data = new_data;
}

/**
 * Zero pads all rows of given flat matrix to a specified length
 *
 * @param data cufftDoubleComplex pointer to the data to be zero padded.
 * @param padded_data Pointer to a cufftDoubleComplex pointer of the location of the new padded data.
 * @param old_data_length Length of data to be padded.
 * @param new_data_length Desired length of data.
 *
 * @return cufftDoubleComplex pointer to new padded data.
 */
void zeroPadMatrixColumns(const cufftDoubleComplex *data, cufftDoubleComplex **padded_data, int old_height, int new_height, int width) {
    // Allocate host memory for new padded data
    cufftDoubleComplex *new_data = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * new_height * width);
    
    memcpy(new_data, data, sizeof(cufftDoubleComplex) * old_height * width);
    memset(new_data + (old_height * width), 0, sizeof(cufftDoubleComplex)*(new_height -  old_height)*width);
    *padded_data = new_data;
}

//------------------------------------------------------------------------------------------------------
// DATA HANDLING METHODS
//------------------------------------------------------------------------------------------------------
/**
 * Reads data from a binary file into a cufftDoubleComplex array.
 *
 * @param host_data_destination cufftDoubleComplex* of the flat array of data into which data from the .bin file is to be read.
 * @param length Length of host_data_destination (i.e. length of fast time * length of slow time).
 * @param filename Input file name (including path details).
 *
 * @return void
 */
void loadDataIntoComplex(cufftDoubleComplex *host_data_destination, size_t data_length, std::string filename) {
    std::cout << "Loading SAR data." << std::endl;
    std::ifstream file(filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate); //opening binary files
    
    if(!file) { std::cout << "Could not open " << filename << "!" << std::endl; }
    
    // If file open, read file
    else {
        // Read into memory block array of chars
        std::streampos length = file.tellg();
        
        char *memblock = new char[length];
        file.seekg (0, std::ios::beg);
        file.read (memblock, length);
        file.close();

        // If double
        double *double_values = (double*) memblock; // reinterpret block of chars as doubles
        
        int i;
        for (i = 0; i < data_length; i++) {
            host_data_destination[i].x = double_values[i];
            host_data_destination[i].y = 0; // no imaginary part for purely double Casie data
        }

        std::cout << i << " double values read." << std::endl;

        // If complex
        // for (int i = 0; i < ((data_length+1)*2); i=i+2) {
        //    host_data_destination[i].x = double_values[i];
        //    host_data_destination.y = double_values[i+1]; // no imaginary part for Casie data. To-do: Correct this!
        // }

        file.close();

        // Clean up memory used to read data
        delete [] double_values;

    }

    std::cout << "Data loaded successfully!\n" << std::endl;
}

/**
 * Reads data from a binary file into a double array.
 *
 * @param host_data_destination double* of the flat array of data into which data from the .bin file is to be read.
 * @param length Length of host_data_destination.
 * @param filename Input file name (including path details).
 *
 * @return void
 */
 void loadDataIntoDouble(double *host_data_destination, size_t data_length, std::string filename) {
    std::cout << "Loading from " << filename << "." << std::endl;
    std::ifstream file(filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate); //opening binary files
    
    if(!file) { std::cout << "Could not open " << filename << "!" << std::endl; }
    
    // If file open, read file
    else {
        // Read into memory block array of chars
        std::streampos length = file.tellg();
        
        char *memblock = new char[length];
        file.seekg (0, std::ios::beg);
        file.read (memblock, length);
        file.close();

        // If double
        double *double_values = (double*) memblock; // reinterpret block of chars as doubles
        
        int i;
        for (i = 0; i < data_length; i++) {
            host_data_destination[i] = double_values[i];
        }

        std::cout << i << " double values read." << std::endl;

        file.close();

        // Clean up memory used to read data
        delete [] double_values;
    }

    std::cout << "Data loaded successfully!\n" << std::endl;
}

/**
 * Writes data contained in a cufftDoubleComplex array to a binary file.
 *
 * @param data cufftDoubleComplex* of the flat array of data to be written to the output file.
 * @param data_length Length of data.
 * @param filename Output file name (including path details).
 *
 * @return void
 */
void writeData(cufftDoubleComplex *data, size_t data_length, std::string filename) {
    std::cout << "Writing data to " << filename << "." << std::endl;
    std::ofstream file(filename, std::ios::binary);
    if(!file) { std::cout << "Could not open " << filename << "!" << std::endl; }

    else {

        file.write((char*) data, sizeof(double) * data_length * 2 );
        
        // for (int i = 0; i < data_length; i++) { 
        //     double real = data[i].x;
        //     double complex = data[i].y;

        //     file.write(reinterpret_cast<char*>(&real), sizeof(double));
        //     file.write(reinterpret_cast<char*>(&complex), sizeof(double));
        // }

        file.close();
    }

    std::cout << "Data written successfully!\n" << std::endl;
}

/**
 * Returns index in a flat 1-D matrix of a given point in a 2-D matrix.
 *
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param width Width of matrix.
 * @param height Height of matrix.
 *
 * @return index in flat 1-D matrix.
 */
__host__ __device__ int getIndex(int x, int y, int width, int height) {
    return (x * width) + y;
}

//------------------------------------------------------------------------------------------------------
// WINDOWING FUNCTIONS
//------------------------------------------------------------------------------------------------------
/**
 * Creates a (rows x cols) matrix of Blackman-Harris windowing functions.
 *
 * @param rows Length of each window function.
 * @param cols Number of duplicate window functions.
 *
 * @return cufftDoubleComplex pointer to generated window function matrix.
 */
 cufftDoubleComplex* blackmanMatrix(int width, int height) { // To-do: Might want to put this in separate file
    // Allocate host memory for window function
    cufftDoubleComplex *window_matrix;
    size_t window_matrix_mem_size = sizeof(cufftDoubleComplex) * width * height;
    window_matrix = (cufftDoubleComplex*) malloc(window_matrix_mem_size);

    for (int i=0; i<width; i++) {
        double blackman_value = 0.42 - 0.5*cos(2.0*PI* i/(width-1)) + 0.08*cos(4.0*PI*i/(width-1));
        for (int j=0; j<height; j++) { // using rows inside nested for loop so that identical value of Blackman window does not have to be recalculated
            int mtrx_pos = getIndex(j, i, width, height); 
            window_matrix[mtrx_pos].x = blackman_value;
            window_matrix[mtrx_pos].y = 0;
        }
    }

    return window_matrix;
}



//------------------------------------------------------------------------------------------------------
// DEBUG METHODS
//------------------------------------------------------------------------------------------------------

cufftDoubleComplex* initialiseSmallMatrix(int width, int height) {
    // Allocate host memory for SAR data
    cufftDoubleComplex *h_small_matrix; // host data
    size_t h_small_matrix_size = sizeof(cufftDoubleComplex) * height * width; // size of dataset
    h_small_matrix = (cufftDoubleComplex*) malloc(h_small_matrix_size);

    for (int i=0; i<(height * width); i++) {
        h_small_matrix[i].x = i;
        h_small_matrix[i].y = 2;        
    }

    return h_small_matrix;
}

void printComplexMatrix(cufftDoubleComplex* matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            std::cout << matrix[getIndex(row, col, width, height)].x << "+" << matrix[getIndex(row, col, width, height)].y << "i  ";
        }
        std::cout << std::endl;
    }
}

void printDouble2Matrix(double2* matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            std::cout << matrix[getIndex(row, col, width, height)].y << " ";// << matrix[getIndex(row, col, width, height)].y << "]  "; // "+" << matrix[getIndex(r, c, width, height)].y << "i  ";
        }
        std::cout << "\n" << std::endl;
    }
}

void printDoubleMatrix(double* matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            std:: cout << matrix[getIndex(row, col, width, height)] << " ";
        }
        std::cout << std::endl;
    }
}

void printIntMatrix(int* matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            std:: cout << matrix[getIndex(row, col, width, height)] << " ";
        }
        std::cout << std::endl;
    }
}