#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
template <typename scalar_t>
__global__ void
group_points_kernel(int b, int c, int n, int npoints, int nsample,
		    const scalar_t *__restrict__ points,
		    const int *__restrict__ idx, scalar_t *__restrict__ out) {
    int batch_index = blockIdx.x;
    points += batch_index * n * c;
    idx += batch_index * npoints * nsample;
    out += batch_index * npoints * nsample * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * npoints; i += stride) {
	const int l = i / npoints;
	const int j = i % npoints;
	for (int k = 0; k < nsample; ++k) {
	    int ii = idx[j * nsample + k];
	    out[(l * npoints + j) * nsample + k] = points[l * n + ii];
	}
    }
}

std::vector<at::Tensor> group_points_cuda(at::Tensor points, at::Tensor idx,
					  at::Tensor output) {
    const int b = points.size(0);
    const int c = points.size(1);
    const int n = points.size(2);
    const int npoints = idx.size(1);
    const int nsample = idx.size(1);

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
	points.type(), "group_points_cuda", ([&] {
	    group_points_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
		b, c, n, npoints, nsample, points.data<scalar_t>(),
		idx.data<int>(), output.data<scalar_t>());
	}));

    return {output};
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
					 int nsample,
					 const scalar_t *__restrict__ grad_out,
					 const int *__restrict__ idx,
					 scalar_t *__restrict__ grad_points) {
    int batch_index = blockIdx.x;
    grad_out += batch_index * npoints * nsample * c;
    idx += batch_index * npoints * nsample;
    grad_points += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * npoints; i += stride) {
	const int l = i / npoints;
	const int j = i % npoints;
	for (int k = 0; k < nsample; ++k) {
	    int ii = idx[j * nsample + k];
	    atomicAdd(grad_points + l * n + ii,
		      grad_out[(l * npoints + j) * nsample + k]);
	}
    }
}

std::vector<at::Tensor> group_points_grad_cuda(at::Tensor grad_out,
					       at::Tensor idx,
					       at::Tensor grad_points) {
    const int b = grad_points.size(0);
    const int c = grad_points.size(1);
    const int n = grad_points.size(2);
    const int npoints = idx.size(1);
    const int nsample = idx.size(1);

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    group_points_grad_kernel<float>
	<<<b, opt_block_config(npoints, c), 0, stream>>>(
	    b, c, n, npoints, nsample, grad_out.data<float>(), idx.data<int>(),
	    grad_points.data<float>());

    return {grad_points};
}
