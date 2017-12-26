#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#ifdef __cplusplus
extern "C" {
#endif

void group_points_kernel_wrapper(int b, int n, int c, int npoints, int nsample,
				 const float *points, const int *idx,
				 float *out, cudaStream_t stream);

void group_points_grad_kernel_wrapper(int b, int n, int c, int npoints,
				      int nsample, const float *grad_out,
				      const int *idx, float *grad_points,
				      cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
