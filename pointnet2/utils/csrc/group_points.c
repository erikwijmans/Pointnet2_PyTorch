#include <THC/THC.h>

#include "group_points_gpu.h"

extern THCState *state;

int group_points_wrapper(int b, int c, int n, int npoints, int nsample,
			 THCudaTensor *points_tensor,
			 THCudaIntTensor *idx_tensor,
			 THCudaTensor *out_tensor) {

    const float *points = THCudaTensor_data(state, points_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);
    float *out = THCudaTensor_data(state, out_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    group_points_kernel_wrapper(b, c, n, npoints, nsample, points, idx, out,
				stream);
    return 1;
}

int group_points_grad_wrapper(int b, int c, int n, int npoints, int nsample,
			      THCudaTensor *grad_out_tensor,
			      THCudaIntTensor *idx_tensor,
			      THCudaTensor *grad_points_tensor) {

    float *grad_points = THCudaTensor_data(state, grad_points_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);
    const float *grad_out = THCudaTensor_data(state, grad_out_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    group_points_grad_kernel_wrapper(b, c, n, npoints, nsample, grad_out, idx,
				     grad_points, stream);
    return 1;
}
