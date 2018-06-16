#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

std::vector<at::Tensor> gather_points_cuda(at::Tensor points, at::Tensor idx,
					   at::Tensor output);
std::vector<at::Tensor> gather_points_grad_cuda(at::Tensor grad_out,
						at::Tensor idx,
						at::Tensor grad_points);
std::vector<at::Tensor> furthest_point_sampling_cuda(int m, at::Tensor dataset,
						     at::Tensor idx);

#endif
