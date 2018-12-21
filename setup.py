from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
import glob

setup(
    name='pointnet2',
    version='0.1',
    author='Erik Wijmans',
    packages=[
        'pointnet2', 'pointnet2/utils', 'pointnet2/utils/pytorch_utils',
        'pointnet2/data', 'pointnet2/models', 'pointnet2/train'
    ],
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=glob.glob('pointnet2/_ext-src/src/*.cpp') +
            glob.glob('pointnet2/_ext-src/src/*.cu'),
            extra_compile_args={
                'cxx': [
                    '-O2', '-I{}'.format(
                        osp.join(
                            osp.dirname(__file__), 'pointnet2/_ext-src/include'))
                ],
                'nvcc': [
                    '-O2', '-I{}'.format(
                        osp.join(
                            osp.dirname(__file__), 'pointnet2/_ext-src/include'))
                ]
            })
    ],
    cmdclass={'build_ext': BuildExtension})
