#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x)                                                          \
    do {                                                                       \
        AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor");             \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                                    \
    do {                                                                       \
        AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor");        \
    } while (0)


