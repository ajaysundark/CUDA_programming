/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned idx = x + y * MATRIX_SIZE;
  if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
    float dotSum = 0.0;
    for (int i=0; i<MATRIX_SIZE; ++i) {
        dotSum+= M.elements[y*MATRIX_SIZE+i]*N.elements[i*MATRIX_SIZE+x];
    }
    P.elements[idx]=dotSum;
  }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
