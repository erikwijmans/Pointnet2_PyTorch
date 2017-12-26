#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
			     const float *known, float *dist2, int *idx,
			     cudaStream_t stream);

void three_interpolate_kernel_wrapper(int b, int m, int c, int n,
				      const float *points, const int *idx,
				      const float *weight, float *out,
				      cudaStream_t stream);

void three_interpolate_grad_kernel_wrapper(int b, int n, int c, int m,
					   const float *grad_out,
					   const int *idx, const float *weight,
					   float *grad_points,
					   cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
