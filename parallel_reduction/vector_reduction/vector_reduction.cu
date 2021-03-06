#ifdef _WIN32
#  define NOMINMAX 
#endif

#define NUM_ELEMENTS 512
#define BLOCK_SIZE 64

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != num_elements)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);
    
    // **===-------- Modify the body of this function -----------===**
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

// Read a floating point vector into M (already allocated) from file
int ReadFile(float* V, char* file_name)
{
    unsigned int data_read = NUM_ELEMENTS;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    for (i = 0; i < data_read; i++) 
        fscanf(input, "%f", &(V[i]));
    return data_read;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of reduction back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
  cudaError_t err = cudaSuccess;
  const unsigned int array_mem_size = sizeof( float) * num_elements;

  // allocate device memory to hold the input and output data
  float* d_data;
  float* o_data;

  err=cudaMalloc((void**)&d_data, array_mem_size);
  if(cudaSuccess != err) {
      printf("Device memory allocation error.");
      exit(0);
  }

  // copy data to the device
  cudaMemcpy(d_data, h_data, array_mem_size, cudaMemcpyHostToDevice);

  int threadsPerBlock = BLOCK_SIZE;
  // because each thread will access two elements, and each block will have two times the block size.
  int blocksPerGrid = ( num_elements + threadsPerBlock - 1 ) / 2*BLOCK_SIZE;
  
  err=cudaMalloc( (void**)&o_data, sizeof(float)*blocksPerGrid );
  if(cudaSuccess != err) {
      printf("Device memory allocation error.");
      exit(0);
  }

/*
  float *temp;
  do {
    reduction<<<blocksPerGrid, threadsPerBlock>>>(d_data, o_data, num_elements);

    num_elements = blocksPerGrid;
    blocksPerGrid = ( num_elements + threadsPerBlock - 1) / 2 * BLOCK_SIZE;
    temp = d_data; d_data = o_data; o_data = temp;
  } while( num_elements > BLOCK_SIZE );
  // get the reduction result back
  //float *resultd = (float *) malloc (num_elements * sizeof(float));
  //cudaMemcpy(&resultd, d_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
*/

  reduction<<<blocksPerGrid, threadsPerBlock>>>(d_data, o_data, num_elements);
  float *resultd = (float *) malloc (blocksPerGrid* sizeof(float));
  cudaMemcpy(resultd, o_data, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i=0; i<num_elements; ++i)
  float result = 0.0f;
  for (int i=0; i<blocksPerGrid; ++i) {
      result += resultd[i];
  }

  cudaFree(d_data);
  cudaFree(o_data);
  free(resultd);

  return result;
}
