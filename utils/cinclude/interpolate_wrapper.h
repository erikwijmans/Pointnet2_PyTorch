

void three_nn_wrapper(int b, int n, int m, THCudaTensor *unknown_tensor,
		      THCudaTensor *known_tensor, THCudaTensor *dist2_tensor,
		      THCudaIntTensor *idx_tensor);
void three_interpolate_wrapper(int b, int c, int m, int n,
			       THCudaTensor *points_tensor,
			       THCudaIntTensor *idx_tensor,
			       THCudaTensor *weight_tensor,
			       THCudaTensor *out_tensor);

void three_interpolate_grad_wrapper(int b, int c, int n, int m,
				    THCudaTensor *grad_out_tensor,
				    THCudaIntTensor *idx_tensor,
				    THCudaTensor *weight_tensor,
				    THCudaTensor *grad_points_tensor);
