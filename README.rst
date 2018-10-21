Pointnet2/Pointnet++ PyTorch
=================

Implemention of `Pointnet2/Pointnet++ <https://github.com/charlesq34/pointnet2>`_ written in `PyTorch <http://pytorch.org>`_.

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
