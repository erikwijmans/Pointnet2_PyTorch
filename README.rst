Pointnet2 PyTorch
=================

Partial implemention of `Pointnet2 <https://github.com/charlesq34/pointnet2>`_ written in `PyTorch <http://pytorch.org>`_.

The custom ops used by Pointnet2 are currently **ONLY** supported on the GPU using CUDA.

Building CUDA kernels
---------------------

- ``mkdir build && cd build``
- ``cmake .. && make``

Exampling training
------------------

Two training examples are provided by ``pointnet2/train/train_sem_seg.py`` and ``train_cls.py``.  The datasets for both will be downloaded automatically by default
