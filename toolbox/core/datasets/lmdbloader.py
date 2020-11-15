from io import BytesIO
import os
import six
import lmdb
#import pyarrow as pa
import torch

from PIL import Image
from torch.utils import data
from caffe2.proto import caffe2_legacy_pb2, caffe2_pb2


def image_decode(bytes):
    byteIO = BytesIO()
    byteIO.write(bytes)
    return Image.open(byteIO)


class ImageLMDB(object):
    """compatible with Caffe LMDB format."""
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.keys = ...
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        caffe_proto = caffe2_legacy_pb2.CaffeDatum()
        caffe_proto = caffe_proto.ParseFromString(byteflow)

        image = image_decode(caffe_proto.data)
        label = caffe_proto.label

        # load image
        imgbuf = caffe_proto[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = caffe_proto[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from torchvision import transforms

    dataset = ImageLMDB(
        '/run/media/mooziisp/仓库/datasets/Kaggle-ILSVRC/ILSVRC/Data/CLS-LOC/val-lmdb',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()])
    )
    print('loaded.')
    print(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16
    )

    ITERITION = 200

    print(f'read data from disk... {len(loader)}')
    record = []
    tik = time.perf_counter()
    for idx, (data, label) in tqdm(enumerate(loader)):
        tok = time.perf_counter()
        record.append(tok - tik)
        if idx == ITERITION:
            break
        if idx == 0:
            print('batch size: ', len(data))
            print('data: ')
            print(data.shape)
            print(data[0], label[0])
    print('test done.')
    print(f'load data: max {max(record):.02f}, min {min(record):.02f}, avg {sum(record)/len(loader):.02f}')
