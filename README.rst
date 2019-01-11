Pointnet2/Pointnet++ PyTorch
============================

* Implemention of Pointnet2/Pointnet++ written in `PyTorch <http://pytorch.org>`_.

* Supports Multi-GPU via `nn.DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_.

* Supports PyTorch version >= 1.0.0.  Use `v1.0 <https://github.com/erikwijmans/Pointnet2_PyTorch/releases/tag/v1.0>`_
  for support of older version of PyTorch.


See the official code release for the paper (in tensorflow), `charlesq34/pointnet2 <https://github.com/charlesq34/pointnet2>`_,
for official model definitions and hyper-parameters.

The custom ops used by Pointnet++ are currently **ONLY** supported on the GPU using CUDA.

Setup
-----

* Install `python3.6` -- This repo is currently only tested with/officially supports `python3.6`

  All further instructions assume that `python` defaults to `python3.6`


* Install dependencies

  ::

    pip install -r requirments.txt


* Building `_ext` module

  ::

    python setup.py build_ext --inplace


Optionally, you can also install this repo as a package via

::

  pip install -e .


Exampling training
------------------

Two training examples are provided by ``pointnet2/train/train_sem_seg.py`` and ``pointnet2/train/train_cls.py``.
The datasets for both will be downloaded automatically by default.


They can be run via

::

  python -m pointnet2.train.train_cls

  python -m pointnet2.train.train_sem_seg


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
