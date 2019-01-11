from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_src_root = 'pointnet2/_ext-src'
_ext_sources = glob.glob('{}/src/*.cpp'.format(_ext_src_root)) + glob.glob(
    '{}/src/*.cu'.format(_ext_src_root))
_ext_headers = glob.glob('{}/include/*'.format(_ext_src_root))

setup(
    name='pointnet2',
    version='2.0',
    author='Erik Wijmans',
    packages=find_packages(),
    install_requires=['etw_pytorch_utils>=1.0', 'h5py'],
    dependency_links=[
        'git+https://github.com/erikwijmans/etw_pytorch_utils.git#egg=etw_pytorch_utils-1.0'
    ],
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
