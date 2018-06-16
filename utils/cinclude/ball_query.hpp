#ifndef BALL_QUERY_HPP
#define BALL_QUERY_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> ball_query(float radius, int nsample, at::Tensor xyz,
                                   at::Tensor new_xyz, at::Tensor idx);

#endif
