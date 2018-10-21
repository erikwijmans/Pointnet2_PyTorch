project(PointNet2)
cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pointnet2/utils")
include_directories("${SOURCE_DIR}/cinclude")
cuda_include_directories("${SOURCE_DIR}/cinclude")
file(GLOB cuda_kernels_src "${SOURCE_DIR}/csrc/*.cu")
cuda_compile(cuda_kernels SHARED ${cuda_kernels_src} OPTIONS -O3)

set(BUILD_CMD python "${SOURCE_DIR}/build_ffi.py")
file(GLOB wrapper_headers "${SOURCE_DIR}/cinclude/*wrapper.h")
file(GLOB wrapper_sources "${SOURCE_DIR}/csrs/*.c")
add_custom_command(OUTPUT "${SOURCE_DIR}/_ext/pointnet2/_pointnet2.so"
		   WORKING_DIRECTORY ${SOURCE_DIR}
		   COMMAND ${BUILD_CMD} --build --objs ${cuda_kernels}
		   DEPENDS ${cuda_kernels}
		   DEPENDS ${wrapper_headers}
		   DEPENDS ${wrapper_sources}
		   VERBATIM)

add_custom_target(pointnet2_ext ALL
		  DEPENDS "${SOURCE_DIR}/_ext/pointnet2/_pointnet2.so")

