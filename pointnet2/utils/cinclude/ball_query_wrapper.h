
int ball_query_wrapper(int b, int n, int m, float radius, int nsample,
		       THCudaTensor *new_xyz_tensor, THCudaTensor *xyz_tensor,
		       THCudaIntTensor *idx_tensor);
