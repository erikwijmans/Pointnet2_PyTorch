__version__ = '2.0'

try:
    __POINTNET2_SETUP__
except NameError:
    __POINTNET2_SETUP__ = False

if not __POINTNET2_SETUP__:
    from pointnet2 import utils
    from pointnet2 import data
    from pointnet2 import models
