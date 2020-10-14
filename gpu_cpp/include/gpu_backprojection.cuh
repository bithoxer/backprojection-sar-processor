/*******************************************************************************************************
*
* Header File for Backprojection SAR Processor for LFM-CW SAR Implemnted in CUDA
* Author: Thomas Gwasira
* Date: 22 September 2020
*
* This is an NVDIA GPU C++ implementation of a Linear Frequency-Modulated
* Continuous-Wave Synthetic Aperture Radar Backprojection image formation routine.
*
********************************************************************************************************/


//------------------------------------------------------------------------------------------------------
// CONSTANTS
//------------------------------------------------------------------------------------------------------

// Transpose
#define BLOCK_DIM 8
#define BLOCK_ROWS 8

// General Constants
# define PI acos(-1.0)
# define c0 299702547                       // speed of light in the atmosphere (m/s)


//------------------------------------------------------------------------------------------------------
// METHOD DECLARATIONS
//------------------------------------------------------------------------------------------------------

__global__ void backprojectionKernel(cufftDoubleComplex *d_img, cufftDoubleComplex *d_range_compressed, int range_compressed_width, int range_compressed_height, double *d_X, double r_n, double *d_Y, double a_n, double z2_n, double betapic, double lambda4pi, double delrsamp, int  N_RANGE_COMPRESSED_LEFT_HALF, double azBW, int grid_length, int n);
// Algorithm Methods
cufftDoubleComplex* loadAndPrepareSARData(void);
cufftDoubleComplex* rangeCompression(cufftDoubleComplex* h_data_windowed_padded );
void loadAndPrepareGeometryData(double **r, double **a, double **z, int geometry_length, std::string filename);
cufftDoubleComplex* indexRangeCompressedData(cufftDoubleComplex *h_range_compressed, int range_compressed_width, int range_compressed_height, int *h_ids, int ids_length, int n);
__global__ void indexRangeCompressedDataKernel(cufftDoubleComplex *d_out, cufftDoubleComplex *d_range_compressed, int range_compressed_width, int range_compressed_height, int *d_ids, int length, int n);
cufftDoubleComplex* multiplyPhase(cufftDoubleComplex *h_phase_intermediate, int *h_az_, double *h_az2_scaled_exp, int length);
__global__ void multiplyPhaseKernel(cufftDoubleComplex *d_out, cufftDoubleComplex *d_phase_intermediate, int *d_az_, double *d_az2_scaled_exp, int length);
void imageFormation(cufftDoubleComplex *range_compressed, int range_compressed_width, int range_compressed_height, double *r, double *a, double *z, int geometry_length);



cufftDoubleComplex* expectedPhase(double *h_in, int length);
__global__ void expectedPhaseKernel(double *d_in, cufftDoubleComplex *d_out, int length);

// Windowing Functions
cufftDoubleComplex* blackmanMatrix(int width, int height); 

// Double Array Operations
double doubleArrayMean(double *h_in, int length);

double* doubleArrayScalarMultiply(double *h_in, int length, double scalar) ;
__global__ void doubleArrayScalarMultiplyKernel(double *d_in, double *d_out, int length, double scalar);

double* doubleArrayScalarAdd(double* h_in, int length, double scalar);
__global__ void doubleArrayScalarAddKernel(double *d_in, double *d_out, int length, double scalar);

double* doubleArrayScalarSubtract(double* h_in, int length, double scalar);
__global__ void doubleArrayScalarSubtractKernel(double *d_in, double *d_out, int length, double scalar);

double* doubleArrayVectorElementwiseMultiply(double *h_in_a, double *h_in_b, int length);
__global__ void doubleArrayVectorElementwiseMultiplyKernel(double *d_in_a, double *d_in_b, double *d_out, int length);

double* doubleArrayVectorAdd(double* h_in_a, double* h_in_b, int length);
__global__ void doubleArrayVectorAddKernel(double *d_in_a, double *d_in_b, double *d_out, int length);

double* doubleArrayVectorSubtract(double* h_in_a, double *h_in_b, int length);
__global__ void doubleArrayVectorSubtractKernel(double *d_in_a, double *d_in_b, double *d_out, int length);

double* doubleArrayElementwiseSquare(double *h_in, int length);
__global__ void doubleArrayElementwiseSquareKernel(double *d_in, double *d_out, int length);

double* doubleArrayElementwiseSquareRoot(double *h_in, int length);
__global__ void doubleArrayElementwiseSquareRootKernel(double *d_in, double *d_out, int length);

double* doubleArraySign(double* h_in, int length);
__global__ void doubleArraySignKernel(double *d_in, double *d_out, int length);

double* doubleArrayVectorAtan2(double *h_in_a, double *h_in_b, int length);
__global__ void doubleArrayVectorAtan2Kernel(double *d_in_a, double *d_in_b, double *d_out, int length);

int* doubleArrayScalarDivide(double *h_in, int length, double scalar);
__global__ void doubleArrayScalarDivideKernel(double *d_in, int *d_out, int length, double scalar);

int* boundaryCorrectIndexes(int* h_in, int length, int N);
__global__ void boundaryCorrectIndexesKernel(int *d_in, int *d_out, int length, int N);

__global__ void compareDoubleArrayToThresholdKernel(double *d_in, int *d_out, int length, double threshold);
int* compareDoubleArrayToThreshold(double* h_in, int length, double threshold);

double* doubleArrayElementwiseExp(double *h_in, int length);
__global__ void doubleArrayElementwiseExpKernel(double *d_in, double *d_out, int length);

double* doubleRange(double start, double stop, double step, int length);

double2* doubleMeshgrid(double *x, double *y, int x_length, int y_length);
double* doubleMeshgridX(double *x, int x_length, int y_length);
double* doubleMeshgridY(double *y, int x_length, int y_length);


// Data Handling Methods
void loadDataIntoComplex(cufftDoubleComplex *host_data_destination, size_t data_length, std::string filename);
void loadDataIntoDouble(double *host_data_destination, size_t data_length, std::string filename);
void writeData(cufftDoubleComplex *data, size_t data_length, std::string filename);
cufftDoubleComplex* initialiseSmallMatrix(int width, int height);
void printComplexMatrix(cufftDoubleComplex* matrix, int width, int height);
void printDouble2Matrix(double2* matrix, int width, int height);
void printDoubleMatrix(double* matrix, int width, int height);
void printIntMatrix(int* matrix, int width, int height);

// Matrix Operations
cufftDoubleComplex* elementWiseComplexMultiply(cufftDoubleComplex *h_in_a, cufftDoubleComplex *h_in_b, int length);
__global__ void elementWiseComplexMultiplyKernel(cufftDoubleComplex* a, cufftDoubleComplex* b, cufftDoubleComplex* ans, int flat_matrix_length);
__global__ void transposeMatrixKernel(cufftDoubleComplex *odata, cufftDoubleComplex *idata, int width, int height);
cufftDoubleComplex* complexZeros(int width, int height);

cufftDoubleComplex* elementWiseComplexAdd(cufftDoubleComplex *h_in_a, cufftDoubleComplex *h_in_b, int length);
__global__ void elementWiseComplexAddKernel(cufftDoubleComplex* d_in_a, cufftDoubleComplex* d_in_b, cufftDoubleComplex* d_out, int length);

__host__ __device__ int getIndex(int x, int y, int width, int height); 
cufftDoubleComplex* transposeMatrix(const cufftDoubleComplex* h_idata, const unsigned int width, const unsigned int height);
void zeroPadMatrixRows(const cufftDoubleComplex *data, cufftDoubleComplex **padded_data, int old_width, int new_width, int height);
void zeroPadMatrixColumns(const cufftDoubleComplex *data, cufftDoubleComplex **padded_data, int old_height, int new_height, int width);
void discardLastMatrixRows(const cufftDoubleComplex *data, cufftDoubleComplex **trimmed_data, int old_width, int new_width, int height);