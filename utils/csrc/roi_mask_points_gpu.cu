#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "roi_mask_points_gpu.h"

// roi format: [w, d, h, theta, cx, cy, cz]
__device__ bool is_in_roi(const float *__restrict__ xyz,
			  const float *__restrict__ roi) {
    const float w = roi[0], d = roi[1], h = roi[2], theta = roi[3], cx = roi[4],
		cy = roi[5], cz = roi[6];
    const float x = xyz[0], y = xyz[1], z = xyz[2];

    const float sinval = sin(theta);
    const float cosval = cos(theta);

    const float bx_x = w * cosval;
    const float bx_y = d * -sinval;

    const float by_x = w * sinval;
    const float by_y = d * cosval;

    const float dx = fabs(x - cx), dy = fabs(y - cy), dz = fabs(z - cz);

    return dx <= fabs(bx_x + by_x) && dy <= fabs(bx_y + by_y) && dz <= h;
}

// Input rois (n_roi, 7), batch_indices (n_roi), data_xyz (b, n, 3)
// Ouput mask (n_roi, n)
__global__ void roi_mask_kernel(int n_roi, int b, int n,
				const float *__restrict__ rois,
				const long *__restrict__ batch_indices,
				const float *__restrict__ data_xyz,
				unsigned char *__restrict__ mask) {

    const int block_idx = blockIdx.x;
    const float *__restrict__ roi = rois + block_idx * 7;
    mask += block_idx * n;

    const long batch_idx = batch_indices[block_idx];
    data_xyz += batch_idx * n * 3;

    const int thread_idx = threadIdx.x;
    const int thread_stride = blockDim.x;
    for (int j = thread_idx; j < n; j += thread_stride) {
	const float *__restrict__ xyz = data_xyz + j * 3;
	mask[j] = is_in_roi(xyz, roi) ? 1 : 0;
    }
}

void roi_mask_kernel_wrapper(int n_roi, int b, int n, const float *rois,
			     const long *batch_indices, const float *data_xyz,
			     unsigned char *mask, cudaStream_t stream) {

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    roi_mask_kernel<<<n_roi, n_threads, 0, stream>>>(
	n_roi, b, n, rois, batch_indices, data_xyz, mask);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}

// Input mask(n_roi, n) batch_indices (n_roi), points (b, n, d)
// Ouput count (n_roi,) descriptors (n_roi, d)
__global__ void roi_avg_pool_kernel_forward(
    int n_roi, int b, int n, int d, const unsigned char *__restrict__ mask,
    const long *__restrict__ batch_indices, const float *__restrict__ points,
    float *__restrict__ descriptors) {

    const int block_idx = blockIdx.x;
    mask += block_idx * n;
    descriptors += block_idx * d;

    const long batch_idx = batch_indices[block_idx];
    points += batch_idx * n * d;

    const int thread_idx = threadIdx.x;
    const int thread_stride = blockDim.x;

    for (int j = thread_idx; j < n; j += thread_stride) {
	if (mask[j] == 1) {
	    for (int c = 0; c < d; ++c) {
		atomicAdd(descriptors + c, points[j * d + c]);
	    }
	}
    }
}

void roi_avg_pool_kernel_forward_wrapper(int n_roi, int b, int n, int d,
					 const unsigned char *mask,
					 const long *batch_indices,
					 const float *points,
					 float *descriptors,
					 cudaStream_t stream) {

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    roi_avg_pool_kernel_forward<<<n_roi, n_threads, 0, stream>>>(
	n_roi, b, n, d, mask, batch_indices, points, descriptors);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}

__global__ void
roi_avg_pool_kernel_backward(int n_roi, int b, int n, int d,
			     const unsigned char *__restrict__ mask,
			     const long *__restrict__ batch_indices,
			     const float *__restrict__ grad_descriptors,
			     float *__restrict__ grad_points) {

    const int block_idx = blockIdx.x;
    mask += block_idx * n;
    grad_descriptors += block_idx * d;

    const long batch_idx = batch_indices[block_idx];
    grad_points += batch_idx * n * d;

    const int thread_idx = threadIdx.x;
    const int thread_stride = blockDim.x;
    for (int j = thread_idx; j < n; j += thread_stride) {
	if (mask[j] == 1) {
	    for (int c = 0; c < d; ++c) {
		atomicAdd(grad_points + j * d + c, grad_descriptors[c]);
	    }
	}
    }
}

void roi_avg_pool_kernel_backward_wrapper(int n_roi, int b, int n, int d,
					  const unsigned char *mask,
					  const long *batch_indices,
					  const float *grad_descriptors,
					  float *grad_points,
					  cudaStream_t stream) {

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    roi_avg_pool_kernel_backward<<<n_roi, n_threads, 0, stream>>>(
	n_roi, b, n, d, mask, batch_indices, grad_descriptors, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}
