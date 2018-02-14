import os.path as osp
import sys
sys.path = [osp.join(osp.dirname(__file__), 'utils')] + sys.path
import pointnet2_modules, pointnet2_utils, pytorch_utils
