#include <THC/THC.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "interpolate_gpu.h"

extern THCState *state;

void three_nn_wrapper(int b, int n, int m, THCudaTensor *unknown_tensor,
		      THCudaTensor *known_tensor, THCudaTensor *dist2_tensor,
		      THCudaIntTensor *idx_tensor) {
    const float *unknown = THCudaTensor_data(state, unknown_tensor);
    const float *known = THCudaTensor_data(state, known_tensor);
    float *dist2 = THCudaTensor_data(state, dist2_tensor);
    int *idx = THCudaIntTensor_data(state, idx_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_nn_kernel_wrapper(b, n, m, unknown, known, dist2, idx, stream);
}

void three_interpolate_wrapper(int b, int c, int m, int n,
			       THCudaTensor *points_tensor,
			       THCudaIntTensor *idx_tensor,
			       THCudaTensor *weight_tensor,
			       THCudaTensor *out_tensor) {

    const float *points = THCudaTensor_data(state, points_tensor);
    const float *weight = THCudaTensor_data(state, weight_tensor);
    float *out = THCudaTensor_data(state, out_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_interpolate_kernel_wrapper(b, c, m, n, points, idx, weight, out,
				     stream);
}

void three_interpolate_grad_wrapper(int b, int c, int n, int m,
				    THCudaTensor *grad_out_tensor,
				    THCudaIntTensor *idx_tensor,
				    THCudaTensor *weight_tensor,
				    THCudaTensor *grad_points_tensor) {

    const float *grad_out = THCudaTensor_data(state, grad_out_tensor);
    const float *weight = THCudaTensor_data(state, weight_tensor);
    float *grad_points = THCudaTensor_data(state, grad_points_tensor);
    const int *idx = THCudaIntTensor_data(state, idx_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_interpolate_grad_kernel_wrapper(b, c, n, m, grad_out, idx, weight,
					  grad_points, stream);
}
