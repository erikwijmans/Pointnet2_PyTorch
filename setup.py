from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

_ext_src_root = 'pointnet2/_ext-src'
_ext_sources = glob.glob('{}/src/*.cpp'.format(_ext_src_root)) + glob.glob(
    '{}/src/*.cu'.format(_ext_src_root))
_ext_headers = glob.glob('{}/include/*'.format(_ext_src_root))

requirements = ['etw_pytorch_utils==1.0.0', 'h5py', 'pprint', 'enum34']

setup(
    name='pointnet2',
    version=pointnet2.__version__,
    author='Erik Wijmans',
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                'cxx':
                ['-O2', '-I{}'.format('{}/include'.format(_ext_src_root))],
                'nvcc':
                ['-O2', '-I{}'.format('{}/include'.format(_ext_src_root))]
            })
    ],
    cmdclass={'build_ext': BuildExtension})
