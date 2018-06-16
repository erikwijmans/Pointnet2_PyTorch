#ifndef GROUP_POINTS_HPP
#define GROUP_POINTS_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> group_points(at::Tensor points, at::Tensor idx,
                                     at::Tensor out);

std::vector<at::Tensor> group_points_grad(at::Tensor grad_out, at::Tensor idx,
                                          at::Tensor grad_points);

#endif
