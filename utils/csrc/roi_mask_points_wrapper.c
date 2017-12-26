#include <THC/THC.h>

#include "roi_mask_points_gpu.h"

extern THCState *state;

int roi_mask_wrapper(int n_roi, int b, int n, THCudaTensor *rois_tensor,
		     THCudaLongTensor *batch_indices_tensor,
		     THCudaTensor *data_xyz_tensor,
		     THCudaByteTensor *mask_tensor) {

	const float *rois = THCudaTensor_data(state, rois_tensor);
	const long *batch_indices =
	    THCudaLongTensor_data(state, batch_indices_tensor);
	const float *data_xyz = THCudaTensor_data(state, data_xyz_tensor);
	unsigned char *mask = THCudaByteTensor_data(state, mask_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);

	roi_mask_kernel_wrapper(n_roi, b, n, rois, batch_indices, data_xyz,
				mask, stream);
	return 1;
}

int roi_avg_pool_forward_wrapper(int n_roi, int b, int n, int d,
				 THCudaByteTensor *mask_tensor,
				 THCudaLongTensor *batch_indices_tensor,
				 THCudaTensor *points_tensor,
				 THCudaTensor *descriptors_tensor) {

	const long *batch_indices =
	    THCudaLongTensor_data(state, batch_indices_tensor);
	const unsigned char *mask = THCudaByteTensor_data(state, mask_tensor);
	const float *points = THCudaTensor_data(state, points_tensor);
	float *descriptors = THCudaTensor_data(state, descriptors_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);
	roi_avg_pool_kernel_forward_wrapper(n_roi, b, n, d, mask, batch_indices,
					    points, descriptors, stream);

	return 1;
}

int roi_avg_pool_backward_wrapper(int n_roi, int b, int n, int d,
				  THCudaByteTensor *mask_tensor,
				  THCudaLongTensor *batch_indices_tensor,
				  THCudaTensor *grad_descriptors_tensor,
				  THCudaTensor *grad_points_tensor) {

	const long *batch_indices =
	    THCudaLongTensor_data(state, batch_indices_tensor);
	const unsigned char *mask = THCudaByteTensor_data(state, mask_tensor);
	const float *grad_descriptors =
	    THCudaTensor_data(state, grad_descriptors_tensor);
	float *grad_points = THCudaTensor_data(state, grad_points_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);
	roi_avg_pool_kernel_backward_wrapper(n_roi, b, n, d, mask,
					     batch_indices, grad_descriptors,
					     grad_points, stream);

	return 1;
}
