#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_
#define BLOCK_SIZE 64
// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, float *partials, int n)
{
  __shared__ float partialSum[2*BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x*2 + threadIdx.x;

  // first add during load thus saving half of threads during the first iteration
  if(i+blockDim.x < n)
    partialSum[tid] = g_data[i] + g_data[i+blockDim.x];
  else
    partialSum[tid]=0;

  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if(tid<s)
        partialSum[tid] += partialSum[tid+s];

    __syncthreads();
  }

  // partial sum from this block to a global memory 
  if(tid==0) partials[blockIdx.x] = partialSum[0];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
