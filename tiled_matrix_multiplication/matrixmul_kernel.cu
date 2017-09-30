/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{

	__shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

	int r = TILE_WIDTH * blockIdx.y + threadIdx.y;
	int c = TILE_WIDTH * blockIdx.x + threadIdx.x;

	float dotSum=0.0;
	for (int i=0; i<(M.width+TILE_WIDTH-1)/TILE_WIDTH; ++i) {
		if (r<M.height && i*TILE_WIDTH+threadIdx.x < M.width)
			Ms[threadIdx.y][threadIdx.x]=M.elements[r*M.width + i*TILE_WIDTH+threadIdx.x];
		else
			Ms[threadIdx.y][threadIdx.x]=0.0;
		if (c<N.width && i*TILE_WIDTH+threadIdx.y < N.height)
			Ns[threadIdx.y][threadIdx.x]=N.elements[(i*TILE_WIDTH+threadIdx.y)*N.width + c];
		else
			Ns[threadIdx.y][threadIdx.x]=0.0;

		__syncthreads();
		for (int j=0; j<TILE_WIDTH; ++j) {
			dotSum+=(Ms[threadIdx.y][j] * Ns[j][threadIdx.x]);
		}
		__syncthreads();
	}

		if (r<P.height && c<P.width)
			P.elements[r*P.width + c] = dotSum;

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
