from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
import glob

setup(
    name='pointnet',
    ext_modules=[
        CUDAExtension(
            name='pointnet',
            sources=[
                'csrc/module_export.cpp',
                'csrc/ball_query.cpp',
                'csrc/ball_query_gpu.cu',
                'csrc/sampling.cpp',
                'csrc/sampling_gpu.cu'
                ,'csrc/group_points.cpp',
                'csrc/group_points_gpu.cu'
            ],
            extra_compile_args={
                'cxx':
                ['-I{}'.format(osp.join(osp.dirname(__file__), 'cinclude'))],
                'nvcc': [
                    '-O2',
                    '-I{}'.format(osp.join(osp.dirname(__file__), 'cinclude'))
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
