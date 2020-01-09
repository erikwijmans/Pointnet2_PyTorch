from setuptools import find_packages, setup

requirements = ["h5py", "pprint", "hydra-core", "pytorch-lightning"]


exec(open("pointnet2/_version.py").read())

setup(
    name="pointnet2",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
)
