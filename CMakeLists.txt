project(PointNet2)
cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/utils/cinclude")
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/utils/cinclude")
file(GLOB cuda_kernels_src "${CMAKE_CURRENT_SOURCE_DIR}/utils/csrc/*.cu")
cuda_compile(cuda_kernels SHARED ${cuda_kernels_src} OPTIONS -O3)

set(BUILD_CMD python "${CMAKE_CURRENT_SOURCE_DIR}/utils/build_ffi.py")
file(GLOB wrapper_headers "${CMAKE_CURRENT_SOURCE_DIR}/utils/cinclude/*wrapper.h")
file(GLOB wrapper_sources "${CMAKE_CURRENT_SOURCE_DIR}/utils/csrs/*.c")
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/utils/_ext/pointnet2/_pointnet2.so"
		   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utils
		   COMMAND ${BUILD_CMD} --build --objs ${cuda_kernels}
		   DEPENDS ${cuda_kernels}
		   DEPENDS ${wrapper_headers}
		   DEPENDS ${wrapper_sources}
		   VERBATIM)

add_custom_target(pointnet2_ext ALL
		  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/utils/_ext/pointnet2/_pointnet2.so")

