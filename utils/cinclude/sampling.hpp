#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> gather_points(at::Tensor points, at::Tensor idx,
                                      at::Tensor out);

std::vector<at::Tensor> gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                                           at::Tensor grad_points);

std::vector<at::Tensor> furthest_point_sampling(int npoint, at::Tensor points,
                                                at::Tensor idx);

#endif
