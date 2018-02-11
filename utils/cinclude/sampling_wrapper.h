
int gather_points_wrapper(int b, int c, int n, int npoints,
			  THCudaTensor *points_tensor,
			  THCudaIntTensor *idx_tensor,
			  THCudaTensor *out_tensor);
int gather_points_grad_wrapper(int b, int c, int n, int npoints,
			       THCudaTensor *grad_out_tensor,
			       THCudaIntTensor *idx_tensor,
			       THCudaTensor *grad_points_tensor);

int furthest_point_sampling_wrapper(int b, int n, int m,
				    THCudaTensor *points_tensor,
				    THCudaTensor *temp_tensor,
				    THCudaIntTensor *idx_tensor);
