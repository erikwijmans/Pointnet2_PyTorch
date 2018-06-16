#include "group_points.hpp"
#include "group_points_gpu.hpp"

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> group_points(at::Tensor points, at::Tensor idx,
				     at::Tensor out) {
    CHECK_INPUT(points);
    CHECK_INPUT(idx);
    CHECK_INPUT(out);

    return group_points_cuda(points, idx, out);
}

std::vector<at::Tensor> group_points_grad(at::Tensor grad_out, at::Tensor idx,
					  at::Tensor grad_points) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(idx);
    CHECK_INPUT(grad_points);

    return group_points_grad_cuda(grad_out, idx, grad_points);
}
