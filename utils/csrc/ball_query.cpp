#include "ball_query.hpp"
#include "ball_query_gpu.h"

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> ball_query(float radius, int nsample, at::Tensor xyz,
                                   at::Tensor new_xyz, at::Tensor idx) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(new_xyz);
    CHECK_INPUT(idx)

    return ball_query_cuda(radius, nsample, xyz, new_xyz, idx);
}
