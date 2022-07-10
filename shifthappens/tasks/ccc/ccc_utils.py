# LMDB code code copied from: https://github.com/rmccorm4/PyTorch-LMDB

import os
import six
import lmdb

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
import pyarrow as pa
import random
import pickle
import itertools

from shifthappens.tasks.ccc.ccc_imagenet_c import noise_transforms


def path_to_dataset(path, root):
    dir_list = []
    for i in range(len(path)):
        dir_list.append(os.path.join(root, "s1_" + str(float(path[i][0]) / 4) + "s2_" + str(float(path[i][1]) / 4)))
    return dir_list

class ApplyTransforms(data.Dataset):
    def __init__(self, data_root, n1, n2, s1, s2, frequency):
        d = noise_transforms()
        self.data_root = data_root
        self.n1 = d[n1]
        self.n2 = d[n2]
        self.s1 = s1
        self.s2 = s2

        self.trn = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224)])
        all_paths = []

        for path, dirs, files in os.walk(self.data_root):
            for name in files:
                all_paths.append(os.path.join(path, name))

        np.random.shuffle(all_paths)
        self.paths = all_paths
        self.paths = self.paths[:frequency]
        all_classes = os.listdir(os.path.join(self.data_root))

        target_list = []
        for cur_path in self.paths:
            cur_class = cur_path.split('/')[-2]
            cur_class = all_classes.index(cur_class)
            target_list.append(cur_class)

        self.targets = target_list

    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            img = Image.open(f)

        img = img.convert('RGB')
        img = self.trn(img)

        if self.s1 > 0:
            img = self.n1(img, self.s1)
            img = Image.fromarray(np.uint8(img))
        if self.s2 > 0:
            img = self.n2(img, self.s2)

        if self.s2 > 0:
            img = Image.fromarray(np.uint8(img))
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        corrupted_img = output.getvalue()
        return corrupted_img, target

    def __len__(self):
        return len(self.paths)



class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def dset2lmdb(dataset, outpath, write_frequency=5000):
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = os.path.expanduser(outpath)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    print("Closing")


if __name__ == "__main__":
    pass
