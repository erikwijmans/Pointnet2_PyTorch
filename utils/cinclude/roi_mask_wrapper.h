
int roi_mask_wrapper(int n_roi, int b, int n, THCudaTensor *rois_tensor,
		     THCudaLongTensor *batch_indices_tensor,
		     THCudaTensor *data_xyz_tensor,
		     THCudaByteTensor *mask_tensor);
int roi_avg_pool_forward_wrapper(int n_roi, int b, int n, int d,
				 THCudaByteTensor *mask_tensor,
				 THCudaLongTensor *batch_indices_tensor,
				 THCudaTensor *points_tensor,
				 THCudaTensor *descriptors_tensor);
int roi_avg_pool_backward_wrapper(int n_roi, int b, int n, int d,
				  THCudaByteTensor *mask_tensor,
				  THCudaLongTensor *batch_indices_tensor,
				  THCudaTensor *grad_descriptors_tensor,
				  THCudaTensor *grad_points_tensor);
