#include "ball_query.hpp"
#include "group_points.hpp"
#include "sampling.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query", &ball_query,
          "Performs a nearest neighbor search for all points in new_xyz wrt "
          "all points in xyz.");
    m.def("gather_points", &gather_points, "Gathers points into a tensor");
    m.def("gather_points_grad", &gather_points_grad,
          "Gathers points into a tensor grad");
    m.def("furthest_point_sampling", &furthest_point_sampling,
          "Performs iterative furthest point samplig");
    m.def("group_points", &group_points, "Groups points into a tensor");
    m.def("group_points_grad", &group_points_grad,
          "Grad Groups points into a tensor");
}
