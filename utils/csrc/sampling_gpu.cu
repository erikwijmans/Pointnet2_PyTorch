#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template <typename scalar_t>
__global__ void gather_points_kernel(int b, int c, int n, int m,
				     const scalar_t *__restrict__ points,
				     const int *__restrict__ idx,
				     scalar_t *__restrict__ out) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
	for (int l = blockIdx.y; l < c; l += gridDim.y) {
	    for (int j = threadIdx.x; j < m; j += blockDim.x) {
		int a = idx[i * m + j];
		out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
	    }
	}
    }
}

std::vector<at::Tensor> gather_points_cuda(at::Tensor points, at::Tensor idx,
					   at::Tensor output) {

    const int b = points.size(0);
    const int c = points.size(1);
    const int m = idx.size(1);
    const int n = points.size(2);

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
	points.type(), "gather_points_cuda", ([&] {
	    gather_points_kernel<scalar_t>
		<<<dim3(b, c, 1), opt_n_threads(m), 0, stream>>>(
		    b, c, n, m, points.data<scalar_t>(), idx.data<int>(),
		    output.data<scalar_t>());
	}));

    return {output};
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
					  const scalar_t *__restrict__ grad_out,
					  const int *__restrict__ idx,
					  scalar_t *__restrict__ grad_points) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
	for (int l = blockIdx.y; l < c; l += gridDim.y) {
	    for (int j = threadIdx.x; j < m; j += blockDim.x) {
		int a = idx[i * m + j];
		atomicAdd(grad_points + (i * c + l) * n + a,
			  grad_out[(i * c + l) * m + j]);
	    }
	}
    }
}

std::vector<at::Tensor> gather_points_grad_cuda(at::Tensor grad_out,
						at::Tensor idx,
						at::Tensor grad_points) {
    const int b = grad_out.size(0);
    const int c = grad_out.size(1);
    const int m = grad_out.size(2);
    const int n = grad_points.size(2);
    grad_points = grad_points.zero_();

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    gather_points_grad_kernel<float>
	<<<dim3(b, c, 1), opt_n_threads(m), 0, stream>>>(
	    b, c, n, m, grad_out.data<float>(), idx.data<int>(),
	    grad_points.data<float>());

    return {grad_points};
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
			 int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp, int *__restrict__ idxs) {
    if (m <= 0)
	return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
	idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
	int besti = 0;
	float best = -1;
	float x1 = dataset[old * 3 + 0];
	float y1 = dataset[old * 3 + 1];
	float z1 = dataset[old * 3 + 2];
	for (int k = tid; k < n; k += stride) {
	    float x2, y2, z2;
	    x2 = dataset[k * 3 + 0];
	    y2 = dataset[k * 3 + 1];
	    z2 = dataset[k * 3 + 2];
	    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
	    if (mag <= 1e-3)
		continue;

	    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
		      (z2 - z1) * (z2 - z1);

	    float d2 = min(d, temp[k]);
	    temp[k] = d2;
	    besti = d2 > best ? k : besti;
	    best = d2 > best ? d2 : best;
	}
	dists[tid] = best;
	dists_i[tid] = besti;
	__syncthreads();

	if (block_size >= 512) {
	    if (tid < 256) {
		__update(dists, dists_i, tid, tid + 256);
	    }
	    __syncthreads();
	}
	if (block_size >= 256) {
	    if (tid < 128) {
		__update(dists, dists_i, tid, tid + 128);
	    }
	    __syncthreads();
	}
	if (block_size >= 128) {
	    if (tid < 64) {
		__update(dists, dists_i, tid, tid + 64);
	    }
	    __syncthreads();
	}
	if (block_size >= 64) {
	    if (tid < 32) {
		__update(dists, dists_i, tid, tid + 32);
	    }
	    __syncthreads();
	}
	if (block_size >= 32) {
	    if (tid < 16) {
		__update(dists, dists_i, tid, tid + 16);
	    }
	    __syncthreads();
	}
	if (block_size >= 16) {
	    if (tid < 8) {
		__update(dists, dists_i, tid, tid + 8);
	    }
	    __syncthreads();
	}
	if (block_size >= 8) {
	    if (tid < 4) {
		__update(dists, dists_i, tid, tid + 4);
	    }
	    __syncthreads();
	}
	if (block_size >= 4) {
	    if (tid < 2) {
		__update(dists, dists_i, tid, tid + 2);
	    }
	    __syncthreads();
	}
	if (block_size >= 2) {
	    if (tid < 1) {
		__update(dists, dists_i, tid, tid + 1);
	    }
	    __syncthreads();
	}

	old = dists_i[0];
	if (tid == 0)
	    idxs[j] = old;
    }
}

std::vector<at::Tensor> furthest_point_sampling_cuda(int m, at::Tensor dataset,
						     at::Tensor idxs) {

    const int b = dataset.size(0);
    const int n = dataset.size(1);
    unsigned int n_threads = opt_n_threads(n);
    auto temp = at::zeros(dataset.type(), {b, n});
    temp.fill_(1e10);

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
	dataset.type(), "furthest_point_sampling_cuda", ([&] {
	    switch (n_threads) {
	    case 512:
		furthest_point_sampling_kernel<scalar_t, 512>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 256:
		furthest_point_sampling_kernel<scalar_t, 256>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 128:
		furthest_point_sampling_kernel<scalar_t, 128>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 64:
		furthest_point_sampling_kernel<scalar_t, 64>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 32:
		furthest_point_sampling_kernel<scalar_t, 32>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 16:
		furthest_point_sampling_kernel<scalar_t, 16>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 8:
		furthest_point_sampling_kernel<scalar_t, 8>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 4:
		furthest_point_sampling_kernel<scalar_t, 4>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 2:
		furthest_point_sampling_kernel<scalar_t, 2>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    case 1:
		furthest_point_sampling_kernel<scalar_t, 1>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
		break;
	    default:
		furthest_point_sampling_kernel<scalar_t, 1>
		    <<<b, n_threads, 0, stream>>>(
			b, n, m, dataset.data<scalar_t>(),
			temp.data<scalar_t>(), idxs.data<int>());
	    }
	}));

    return {idxs};
}
