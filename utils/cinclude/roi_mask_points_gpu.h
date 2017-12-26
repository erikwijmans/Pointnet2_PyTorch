
#ifndef _ROI_MASK_POINTS_GPU_H
#define _ROI_MASK_POINTS_GPU_H

#ifdef __cplusplus
extern "C" {
#endif
void roi_mask_kernel_wrapper(int n_roi, int b, int n, const float *rois,
			     const long *batch_indices, const float *data_xyz,
			     unsigned char *mask, cudaStream_t stream);

void roi_avg_pool_kernel_forward_wrapper(int n_roi, int b, int n, int d,
					 const unsigned char *mask,
					 const long *batch_indices,
					 const float *points,
					 float *descriptors,
					 cudaStream_t stream);

void roi_avg_pool_kernel_backward_wrapper(int n_roi, int b, int n, int d,
					  const unsigned char *mask,
					  const long *batch_indices,
					  const float *grad_descriptors,
					  float *grad_points,
					  cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
