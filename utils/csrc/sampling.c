#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;

int gather_points_wrapper(int b, int c, int n, int npoints,
			  THCudaTensor *points_tensor,
			  THCudaIntTensor *idx_tensor,
			  THCudaTensor *out_tensor) {

    const float *points = THCudaTensor_data(state, points_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);
    float *out = THCudaTensor_data(state, out_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    gather_points_kernel_wrapper(b, c, n, npoints, points, idx, out, stream);
    return 1;
}

int gather_points_grad_wrapper(int b, int c, int n, int npoints,
			       THCudaTensor *grad_out_tensor,
			       THCudaIntTensor *idx_tensor,
			       THCudaTensor *grad_points_tensor) {

    const float *grad_out = THCudaTensor_data(state, grad_out_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);
    float *grad_points = THCudaTensor_data(state, grad_points_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    gather_points_grad_kernel_wrapper(b, c, n, npoints, grad_out, idx,
				      grad_points, stream);
    return 1;
}

int furthest_point_sampling_wrapper(int b, int n, int m,
				    THCudaTensor *points_tensor,
				    THCudaTensor *temp_tensor,
				    THCudaIntTensor *idx_tensor) {

    const float *points = THCudaTensor_data(state, points_tensor);
    float *temp = THCudaTensor_data(state, temp_tensor);
    int *idx = THCudaIntTensor_data(state, idx_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    furthest_point_sampling_kernel_wrapper(b, n, m, points, temp, idx, stream);
    return 1;
}
