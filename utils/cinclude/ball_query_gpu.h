#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#include <torch/torch.h>

std::vector<at::Tensor> ball_query_cuda(float radius, int nsample,
					at::Tensor xyz, at::Tensor new_xyz,
					at::Tensor idx);

#endif
