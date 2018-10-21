#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#ifdef __cplusplus
extern "C" {
#endif

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
				     int nsample, const float *xyz,
				     const float *new_xyz, int *idx,
				     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
