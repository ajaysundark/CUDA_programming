#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define TILE_SIZE 1024
#define BLOCK_SIZE 256

unsigned int **sBlockSums;
unsigned int **rBlockSums;
unsigned int *sizeArray;
unsigned int n=0; // no of blocksums

// Memory allocation at device
void preallocBlockSums(int num_elements) {
	n = ceil(log(num_elements)/log(2*BLOCK_SIZE));
	sBlockSums = (unsigned int **) malloc(sizeof(unsigned int *)* n);
	rBlockSums = (unsigned int **) malloc(sizeof(unsigned int *)* n);
	sizeArray = (unsigned int *) malloc(sizeof(unsigned int) * n);

	for(int i=0; i<n; ++i) {
		num_elements = ceil((float)num_elements/(2*BLOCK_SIZE));
		if(num_elements == 0)
			num_elements = 1;

		sizeArray[i] = num_elements;

		cudaMalloc(&sBlockSums[i], sizeof(unsigned int)*num_elements);
		cudaMalloc(&rBlockSums[i], sizeof(unsigned int)*num_elements);
	}
}

// Memory cleanup at device

void deallocBlockSums()
{
	for(int i=0; i<n; i++)
	{
		cudaFree(sBlockSums[i]);
		cudaFree(rBlockSums[i]);
		free(sizeArray);
	}

	free(sBlockSums);
	free(rBlockSums);
}

// Kernel Functions

__global__ void sum_kernel(unsigned int *outp, unsigned int *inpp, unsigned int n)
{
	if(blockIdx.x > 0)
	{
		int tid = threadIdx.x;
		int start = blockIdx.x * blockDim.x * 2;

		outp[start+tid] += inpp[blockIdx.x];
		outp[start+blockDim.x+tid] += inpp[blockIdx.x];
	}
}

__global__ void scan_kernel(unsigned int *outp, unsigned int *inp, unsigned int *blockSum, unsigned int num_elements)
{
	__shared__ unsigned int scan_array[2*BLOCK_SIZE];

	int tx = threadIdx.x;
	int startIdx = 2 * blockIdx.x * blockDim.x;

	// copy data into shared memory
	if((startIdx+tx) < num_elements)
		scan_array[tx] = inp[startIdx + tx];
	else
		scan_array[tx] = 0;

	if((startIdx+blockDim.x+tx) < num_elements)
		scan_array[blockDim.x + tx] = inp[startIdx + blockDim.x + tx];
	else
		scan_array[blockDim.x + tx] = 0;

	__syncthreads();

	// reduction
	int stride=1;
	while(stride <= BLOCK_SIZE)
	{
	    int index = (threadIdx.x+1)*stride*2 - 1;
	    if(index < 2*BLOCK_SIZE)
	        scan_array[index] += scan_array[index-stride];
	    stride = stride*2;
	    __syncthreads();
	}
	
	if (threadIdx.x==0) // do this only for the first time
	{ 
		blockSum[blockIdx.x] = scan_array[2*blockDim.x-1];
        	scan_array[2*blockDim.x-1] = 0;
	}

    // post-scan
	stride = BLOCK_SIZE;
	while(stride > 0) 
	{
	   int index = (threadIdx.x+1)*stride*2 - 1;
	   if(index < 2* BLOCK_SIZE) 
	   {
		//swap
	      unsigned int temp = scan_array[index];
	      scan_array[index] += scan_array[index-stride]; 
	      scan_array[index-stride] = temp; 
	   } 
	   stride = stride / 2;
	   __syncthreads(); 
	} 

	// copy the scan result in shared memory to output array
	if((startIdx+tx) < num_elements)
		outp[startIdx + tx] = scan_array[tx];
	if((startIdx+blockDim.x+tx) < num_elements)
		outp[startIdx + blockDim.x + tx] = scan_array[blockDim.x + tx];
}


// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE


void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
	int grid = sizeArray[0];
	int threads = BLOCK_SIZE;

	// first scan
    scan_kernel<<<grid,threads>>>(outArray, inArray, sBlockSums[0], numElements);
    
    int level=0;
	while(grid > 1)
	{
		grid = sizeArray[level+1];
		scan_kernel<<<grid, threads>>>(rBlockSums[level], sBlockSums[level], sBlockSums[level+1], sizeArray[level]);
		level++;
	}

	// sum scan
	for(int i=level-1; i>=0; --i)
	{
		grid = sizeArray[i+1];
		sum_kernel<<<grid, threads>>>(rBlockSums[i], rBlockSums[i+1], sizeArray[i]);
	}

	grid = sizeArray[0];
	sum_kernel<<<grid, threads>>>(outArray, rBlockSums[0], numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
