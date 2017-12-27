import glob
import torch
from os import path
from torch.utils.ffi import create_extension
import sys

base_dir = path.dirname(path.abspath(__file__))
extra_objects = sys.argv[1:]
extra_objects += [a for a in glob.glob('/usr/local/cuda/lib64/*.a')]

ffi = create_extension(
    '_ext',
    headers=[a for a in glob.glob("cinclude/*_wrapper.h")],
    sources=[a for a in glob.glob("csrc/*.c")],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[path.join(base_dir, 'cinclude')],
    verbose=False)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Needs CUDA!"
    ffi.build()
