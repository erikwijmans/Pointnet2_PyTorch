from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
import glob

_ext_src_root = 'pointnet2/_ext-src'

setup(
    name='pointnet2',
    version='2.0',
    author='Erik Wijmans',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=glob.glob('{}/src/*.cpp'.format(_ext_src_root)) + glob.glob(
                '{}/src/*.cu'.format(_ext_src_root)),
            extra_compile_args={
                'cxx': [
                    '-O2', '-I{}'.format(
                        osp.join(
                            osp.dirname(__file__),
                            '{}/include'.format(_ext_src_root)))
                ],
                'nvcc': [
                    '-O2', '-I{}'.format(
                        osp.join(
                            osp.dirname(__file__),
                            '{}/include'.format(_ext_src_root)))
                ]
            })
    ],
    cmdclass={'build_ext': BuildExtension})
