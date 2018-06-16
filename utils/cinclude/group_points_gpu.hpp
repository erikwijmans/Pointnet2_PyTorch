#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> group_points_cuda(at::Tensor points, at::Tensor idx,
					  at::Tensor output);

std::vector<at::Tensor> group_points_grad_cuda(at::Tensor grad_out,
					       at::Tensor idx,
					       at::Tensor grad_points);
#endif
