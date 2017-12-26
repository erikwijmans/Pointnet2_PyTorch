#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;

int gather_points_wrapper(int b, int n, int c, int npoints,
			  THCudaTensor *points_tensor,
			  THCudaIntTensor *idx_tensor,
			  THCudaTensor *out_tensor) {

	const float *points = THCudaTensor_data(state, points_tensor);
	const int *idx = THCudaIntTensor_data(state, idx_tensor);
	float *out = THCudaTensor_data(state, out_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);

	gather_points_kernel_wrapper(b, n, c, npoints, points, idx, out,
				     stream);
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

	furthest_point_sampling_kernel_wrapper(b, n, m, points, temp, idx,
					       stream);
	return 1;
}
