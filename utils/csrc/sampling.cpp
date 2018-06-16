#include "sampling.hpp"
#include "sampling_gpu.hpp"

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> gather_points(at::Tensor points, at::Tensor idx,
				      at::Tensor out) {

    return gather_points_cuda(points, idx, out);
}

std::vector<at::Tensor> gather_points_grad(at::Tensor grad_out, at::Tensor idx,
					   at::Tensor grad_points) {
    return gather_points_grad_cuda(grad_out, idx, grad_points);
}

std::vector<at::Tensor> furthest_point_sampling(int npoint, at::Tensor points,
						at::Tensor idx) {
    return furthest_point_sampling_cuda(npoint, points, idx);
}
