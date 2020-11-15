"""
https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
"""

import io
import os
import glob

import lmdb
# import pyarrow as pa # DETRACTED
import numpy as np
from PIL import Image
from caffe2.proto import caffe2_legacy_pb2, caffe2_pb2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


DUMMY_LABEL = -1


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


# def dumps_pyarrow(obj):
#     """
#     Serialize an object.
#     Returns:
#         Implementation-dependent bytes-like object
#     """
#     return pa.serialize(obj).to_buffer()


def dump_image_bytes(np_arr, mode="RGB", format="JPEG"):
    im = Image.fromarray(np_arr, mode=mode)

    im_byte = io.BytesIO()
    im.save(im_byte, format=format)
    return im_byte.getvalue()


def make_proto(image, label, format):
    label = label if label else DUMMY_LABEL

    if format == 'caffe':
        caffe_proto = caffe2_legacy_pb2.CaffeDatum()
        # (caffe_proto.height,
        #  caffe_proto.width,
        #  caffe_proto.channels) = im_arr.shape
        caffe_proto.data = image # dump_image_bytes(im_arr, mode="RGB",
                                 #            format="JPEG")  # bytes(img_data.reshape(np.prod(img_data.shape)))
        caffe_proto.label = label
        caffe_proto.encoded = True
    elif format == 'caffe2':
        # Create TensorProtos
        caffe_proto = caffe2_pb2.TensorProtos()

        img_tensor = caffe_proto.protos.add()
        # img_tensor.dims.extend(im_arr.shape)
        # Refs: https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2.proto
        img_tensor.data_type = 3
        img_tensor.byte_data = image # dump_image_bytes(im_arr, mode="RGB", format="JPEG")

        label_tensor = caffe_proto.protos.add()
        label_tensor.data_type = 2  # INT
        label_tensor.int32_data.append(label)
    else:
        raise ValueError

    return caffe_proto


class RawFolder(object):
    """No Label for read raw data from file.
       Only for val and test dataset."""
    def __init__(self, root, split='val'):
        self.root = root
        self.filepaths = glob.glob(os.path.join(root, split) + '/*')
    def __getitem__(self, index):
        key = os.path.relpath(self.filepaths[index], self.root)
        with open(self.filepaths[index], 'rb') as f:
            raw = f.read()
        return key, raw
    def __len__(self):
        return len(self.filepaths)


def folder_to_lmdb(
        dpath,
        output,
        split="train",
        format="caffe",
        label=True,
        map_size=1<<40,
        num_workers=8,
        write_frequency=5000) -> None:
    """Convert image from folder to lmdb format, compatible with
    CaffeReader from NVIDIA DALI.

    :param dpath: point to path in which images stored.
    :param output: point to path in which lmdb file will store.
    :param split: dataset type: train, test, val
    :param format:
    :param label:
    :param map_size: lmdb maximum storage.
    :param num_workers: number of thread to read data from dataset.
    :return: None
    """
    # Refs: https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2_legacy.proto
    directory = os.path.expanduser(os.path.join(dpath, split))
    print(f"Loading dataset from {directory}")
    if label:
        dataset = ImageFolder(directory, loader=raw_reader)
    else:
        dataset = RawFolder(dpath, split)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x)
    print(f"Generate LMDB to {output}")
    db = lmdb.open(output, map_size=map_size)

    with db.begin(write=True) as txn:
        for idx, data in enumerate(dataloader):
            if label:
                image, label = data[0]
                caffe_proto = make_proto(image, label, format)
            else:
                datas = data[0]
                caffe_proto = make_proto(datas[1], None, format)

            txn.put(
                u'{}'.format(idx).encode('ascii'),
                caffe_proto.SerializeToString()  # dumps_pyarrow((image, label))
            )

            if idx and idx % write_frequency == 0:
                print("Inserted {} rows".format(idx))
                break

        print("Flushing database ...")

    db.sync()
    db.close()
    print('done.')


# def test_coding(files):
#     for f in files:
#         env = lmdb.open(f)
#
#         with env.begin() as txn:
#             ks = [ k for k, _ in txn.cursor()]
#             vs = [ v for _, v in txn.cursor()]
#
#         tensor_proto1 = caffe2_pb2.TensorProtos()
#         tensor_proto2 = caffe2_legacy_pb2.CaffeDatum()
#
#         ret1 = tensor_proto1.ParseFromString(vs[0])
#         ret2 = tensor_proto2.ParseFromString(vs[0])
#
#         if ret1 and tensor_proto1.__str__():
#             print(f'{f} is caffe2 format.')
#         if ret2 and tensor_proto2.__str__():
#             print(f'{f} is caffe2 legacy format.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default='val')
    parser.add_argument('-o', '--output', type=str, default='.')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--format', type=str, default='caffe')
    parser.add_argument('-p', '--procs', type=int, default=1)

    args = parser.parse_args()

    folder_to_lmdb(args.folder, args.output,
                   split=args.split, format=args.format,
                   label=args.label, num_workers=args.procs, )

