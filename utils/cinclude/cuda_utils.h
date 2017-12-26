#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

inline int opt_n_threads(int work_size) {
	unsigned int n_threads = work_size;
	n_threads--;
	n_threads |= n_threads >> 1;
	n_threads |= n_threads >> 2;
	n_threads |= n_threads >> 4;
	n_threads |= n_threads >> 8;
	n_threads |= n_threads >> 16;
	n_threads++;

	return max(min(n_threads / 2, 512), 2);
}

#ifdef __cplusplus
}
#endif
#endif
