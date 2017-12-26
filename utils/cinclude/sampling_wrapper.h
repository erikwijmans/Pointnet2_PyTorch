
int gather_points_wrapper(int b, int n, int c, int npoints,
			  THCudaTensor *points_tensor,
			  THCudaIntTensor *idx_tensor,
			  THCudaTensor *out_tensor);

int furthest_point_sampling_wrapper(int b, int n, int m,
				    THCudaTensor *points_tensor,
				    THCudaTensor *temp_tensor,
				    THCudaIntTensor *idx_tensor);
