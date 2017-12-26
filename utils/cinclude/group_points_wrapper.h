
int group_points_wrapper(int b, int n, int c, int npoints, int nsample,
			 THCudaTensor *points_tensor,
			 THCudaIntTensor *idx_tensor, THCudaTensor *out);
int group_points_grad_wrapper(int b, int n, int c, int npoints, int nsample,
			      THCudaTensor *grad_out_tensor,
			      THCudaIntTensor *idx_tensor,
			      THCudaTensor *grad_points_tensor);
