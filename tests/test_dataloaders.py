from pointnet2 import data

def test_modelnet40():
    dset = data.ModelNet40Cls(1024)
    dset[0]

def test_semseg():
    dset = data.Indoor3DSemSeg(1024)
    dset[0]
