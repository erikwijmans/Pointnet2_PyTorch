project(PointNet2)
cmake_minimum_required(VERSION 3.5)

find_package(CUDA)

include_directories("${CMAKE_SOURCE_DIR}/cinclude")
cuda_include_directories("${CMAKE_SOURCE_DIR}/cinclude")
file(GLOB cuda_kernels_src "csrc/*.cu")
cuda_compile(cuda_kernels SHARED ${cuda_kernels_src} OPTIONS -O3)

file(GLOB wrapper_headers "cinclude/*wrapper.h")
add_custom_command(OUTPUT "${CMAKE_SOURCE_DIR}/_ext/__ext.so"
		   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		   COMMAND python "${CMAKE_SOURCE_DIR}/build_ffi.py" ${cuda_kernels}
		   DEPENDS ${cuda_kernels}
		   DEPENDS ${wrapper_headers}
		   VERBATIM)

add_custom_target(ext ALL
	          DEPENDS "${CMAKE_SOURCE_DIR}/_ext/__ext.so")
