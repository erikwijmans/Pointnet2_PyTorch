Pointnet2/Pointnet++ PyTorch
============================

Implemention of Pointnet2/Pointnet++ written in `PyTorch <http://pytorch.org>`_.

Official code release for the paper (in tensorflow): https://github.com/charlesq34/pointnet2

The custom ops used by Pointnet++ are currently **ONLY** supported on the GPU using CUDA.

Building CUDA kernels
---------------------

- ``mkdir build && cd build``
- ``cmake .. && make``

Exampling training
------------------

Two training examples are provided by ``pointnet2/train/train_sem_seg.py`` and ``pointnet2/train/train_cls.py``.  The datasets for both will be downloaded automatically by default.

The scripts expect that you are in the root directory and have that directory added to your ``PYTHONPATH``,
i.e ``export PYTHONPATH=$(pwd):${PYTHONPATH}``


Citation
--------

::

  @article{pytorchpointnet++,
        Author = {Erik Wijmans},
        Title = {Pointnet++ Pytorch},
        Journal = {https://github.com/erikwijmans/Pointnet2_PyTorch},
        Year = {2018}
  }

  @inproceedings{qi2017pointnet++,
      title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
      author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5099--5108},
      year={2017}
  }
