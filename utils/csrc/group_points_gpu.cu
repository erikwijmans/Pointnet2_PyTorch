#include <stdio.h>
#include <stdlib.h>

#include "group_points_gpu.h"
#include "cuda_utils.h"

// input: points(b, n, c) idx(b, npoints, nsample)
// output: out(b, npoints, nsample, c)
__global__ void group_points_kernel(int b, int n, int c, int npoints,
				    int nsample,
				    const float  *__restrict__ points,
				    const int  *__restrict__ idx,
				    float  *__restrict__ out) {
	int batch_index = blockIdx.x;
	points += batch_index * n * c;
	idx += batch_index * npoints * nsample;
	out += batch_index * npoints * nsample * c;

	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int j = index; j < npoints; j += stride) {
		for (int k = 0; k < nsample; ++k) {
			int ii = idx[j * nsample + k];
			memcpy(out + j * nsample * c + k * c, points + ii * c,
			       sizeof(float) * c);
		}
	}
}

void group_points_kernel_wrapper(int b, int n, int c, int npoints, int nsample,
				 const float *points, const int *idx,
				 float *out, cudaStream_t stream) {

	cudaError_t err;
	group_points_kernel<<<b, opt_n_threads(npoints), 0, stream>>>(
	    b, n, c, npoints, nsample, points, idx, out);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n",
			cudaGetErrorString(err));
		exit(-1);
	}
}

// input: grad_out(b, npoints, nsample, c), idx(b, npoints, nsample)
// output: grad_points(b, n, c)
__global__ void group_points_grad_kernel(int b, int n, int c, int npoints,
					 int nsample,
					 const float  *__restrict__ grad_out,
					 const int  *__restrict__ idx,
					 float  *__restrict__ grad_points) {
	int batch_index = blockIdx.x;
	grad_points += batch_index * n * c;
	idx += batch_index * npoints * nsample;
	grad_out += batch_index * npoints * nsample * c;

	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int j = index; j < npoints; j += stride) {
		for (int k = 0; k < nsample; ++k) {
			int ii = idx[j * nsample + k];
			for (int l = 0; l < c; ++l) {
				atomicAdd(
				    grad_points + ii * c + l,
				    grad_out[j * nsample * c + k * c + l]);
			}
		}
	}
}

void group_points_grad_kernel_wrapper(int b, int n, int c, int npoints,
				      int nsample, const float *grad_out,
				      const int *idx, float *grad_points,
				      cudaStream_t stream) {
	cudaError_t err;
	group_points_grad_kernel<<<b, opt_n_threads(npoints), 0, stream>>>(
	    b, n, c, npoints, nsample, grad_out, idx, grad_points);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n",
			cudaGetErrorString(err));
		exit(-1);
	}
}
