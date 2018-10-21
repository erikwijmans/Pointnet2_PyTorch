import glob
import torch
import os.path as osp
from torch.utils.ffi import create_extension
import sys, argparse, shutil

base_dir = osp.dirname(osp.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for building pointnet2 ffi extension"
    )
    parser.add_argument("--objs", nargs="*")
    clean_arg = parser.add_mutually_exclusive_group()
    clean_arg.add_argument("--build", dest='build', action="store_true")
    clean_arg.add_argument("--clean", dest='clean', action="store_true")
    parser.set_defaults(build=False, clean=False)

    args = parser.parse_args()
    assert args.build or args.clean

    return args


def build(args):
    extra_objects = args.objs
    extra_objects += [a for a in glob.glob('/usr/local/cuda/lib64/*.a')]

    ffi = create_extension(
        '_ext.pointnet2',
        headers=[a for a in glob.glob("cinclude/*_wrapper.h")],
        sources=[a for a in glob.glob("csrc/*.c")],
        define_macros=[('WITH_CUDA', None)],
        relative_to=__file__,
        with_cuda=True,
        extra_objects=extra_objects,
        include_dirs=[osp.join(base_dir, 'cinclude')],
        verbose=False,
        package=False
    )
    ffi.build()


def clean(args):
    shutil.rmtree(osp.join(base_dir, "_ext"))


if __name__ == "__main__":
    args = parse_args()
    if args.clean:
        clean(args)
    else:
        build(args)
