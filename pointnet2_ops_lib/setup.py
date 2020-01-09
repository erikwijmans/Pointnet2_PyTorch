import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_src_root = "pointnet2_ops/_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = []

execfile("pointnet2_ops/_version.py")

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
