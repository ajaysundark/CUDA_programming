#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"

uint32_t *d_data;
uint32_t *dup_out;
uint32_t *o_data;


__global__ void truncate_kernel(uint32_t *input, uint32_t dim) {
  if(threadIdx.x<dim && input[threadIdx.x]>UINT8_MAXIMUM)
    input[threadIdx.x]=UINT8_MAXIMUM;
}

__global__ void histogram_kernel(uint32_t *input, uint32_t *output, uint32_t ip_dim, uint32_t op_dim) {
    __shared__ uint32_t hist[HISTO_WIDTH * HISTO_HEIGHT];

    if(threadIdx.x < op_dim)
      hist[threadIdx.x]=0;

    __syncthreads();

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(id<ip_dim) {
      atomicAdd( &( hist[input[id]] ), 1);
      id+=stride;
    }

    __syncthreads();

    uint32_t val = hist[threadIdx.x];
    if(threadIdx.x < op_dim && val>0) {
      uint32_t *add = output+threadIdx.x;
      atomicAdd( add, val);
    }
}

void opt_2dhisto() {
    int threadsPerBlock = NUM_BINS;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock -1)/threadsPerBlock;

    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, dup_out, NUM_ELEMENTS, NUM_BINS);
    cudaDeviceSynchronize();

    truncate_kernel<<<blocksPerGrid, threadsPerBlock>>>(dup_out, NUM_BINS);
    cudaDeviceSynchronize();
}

/* Include below the implementation of any other functions you need */


void pre_alloc_device(uint32_t *data) {
    if(cudaSuccess!=cudaMalloc( &d_data, sizeof(uint32_t) * NUM_ELEMENTS )) {
       printf("device memory allocation error ");
       exit(0);
    }
    cudaMemcpy( d_data, data, sizeof(uint32_t) * NUM_ELEMENTS, cudaMemcpyHostToDevice );

    if(cudaSuccess!=cudaMalloc( &dup_out, sizeof(uint32_t)* NUM_BINS)) {
       printf("device memory allocation error ");
       exit(0);
    }
    cudaMemset( dup_out, 0, sizeof(uint32_t) * NUM_BINS);

    o_data = (uint32_t *) malloc( sizeof(uint32_t) * NUM_BINS);
    memset( o_data, 0, sizeof(uint32_t) * NUM_BINS);
}

void dealloc_device(uint8_t *kernel_bins) {
    cudaMemcpy( o_data, dup_out,  sizeof(uint32_t) * NUM_BINS, cudaMemcpyDeviceToHost );

    for(int k=0; k<NUM_BINS; ++k)
	kernel_bins[k] = o_data[k];

    free(o_data);
    cudaFree(dup_out);
    cudaFree(d_data);
}


void test_device(uint32_t *input[], uint8_t *kernel_bins) {
    uint32_t *data = new uint32_t[NUM_ELEMENTS];
    data_flatten(input, data, INPUT_HEIGHT, INPUT_WIDTH);

    pre_alloc_device(data);
    dealloc_device(kernel_bins);
}


