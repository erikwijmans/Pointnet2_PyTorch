from setuptools import find_packages, setup

requirements = ["h5py", "pprint", "enum34", "future"]


execfile("pointnet2/_version.py")

setup(
    name="pointnet2",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
)
