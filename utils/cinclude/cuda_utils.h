#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, 512), 32);
}

#endif
