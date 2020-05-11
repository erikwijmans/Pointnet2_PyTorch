import os
import os.path as osp
import shlex
import shutil
import subprocess

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40Cls(data.Dataset):
    def __init__(self, num_points, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms

        self.set_num_points(num_points)
        self._cache = os.path.join(BASE_DIR, "modelnet40_normal_resampled_cache")

        if not osp.exists(self._cache):
            self.folder = "modelnet40_normal_resampled"
            self.data_dir = os.path.join(BASE_DIR, self.folder)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )

            if download and not os.path.exists(self.data_dir):
                zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
                subprocess.check_call(
                    shlex.split("curl {} -o {}".format(self.url, zipfile))
                )

                subprocess.check_call(
                    shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
                )

                subprocess.check_call(shlex.split("rm {}".format(zipfile)))

            self.train = train
            self.set_num_points(num_points)

            self.catfile = os.path.join(self.data_dir, "modelnet40_shape_names.txt")
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_train.txt")
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_test.txt")
                        )
                    ]

                shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".txt",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                    osp.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.datapath)):
                        fn = self.datapath[i]
                        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                        cls = self.classes[self.datapath[i][0]]
                        cls = int(cls)

                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(pc=point_set, lbl=cls), use_bin_type=True
                            ),
                        )

            shutil.rmtree(self.data_dir)

        self._lmdb_file = osp.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        point_set = ele["pc"]

        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        point_set = point_set[pt_idxs, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        return point_set, ele["lbl"]

    def __len__(self):
        return self._len

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40Cls(16, train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
